import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from pandas import Series, DataFrame
# from visualise import route_visualise
# from data_process import data_pre


class SubProblem:
    def __init__(self, pi: Series, Vs: set or list[str], Ve: set or list[str], m_f: Series, m_g: Series,
                 s, Se: set or list[str], Q, Q_0, mu, sigma) -> None:
        self.Vs = Vs #订单起点集合
        self.Vs_all = [f"{node}{i}" for node in Vs for i in range(1, 6)]
        self.Ve = Ve #订单终点集合

        self.s = s #指定车库为起点
        self.Se = Se #换电站集合
        self.V_all = list(set(self.Vs_all)|s|Se)  # 所有需要节点集合例如A1,A2,A3,C1,C2,C3,S,s,E,F
        self.Q = Q  # 最大电池容量
        self.Q_0 = Q_0  # 换电站更换的单节电池容量
        self.pi = pi #RMP问题得到的执行满载路段的对偶值
        M = 1000 # M 一个极大的数
        self.m_f = m_f # m_f满载路径f耗电量
        self.m_g = m_g # m_g空载路径g耗电量
        #生成两点之间的耗电量
        q = pd.DataFrame(0.0, index=self.V_all, columns=self.V_all)
        for i in self.V_all:
            for j in self.V_all:
                if i != j:
                    if i[0] in Vs:
                        # 为 m_f 找到以 i[0] 开头的键
                        key_f = next((k for k in self.m_f.keys() if k.startswith(i[0])), None)
                        # 为 m_g 找到以 j 结尾的键
                        key_g = next((k for k in self.m_g.keys() if k.startswith(key_f[1]) and k.endswith(j[0])), None)

                        # 从 m_f 和 m_g 中获取对应的电量消耗值
                        consumption_f = self.m_f.get(key_f, 0)  # 如果键不存在，返回0
                        consumption_g = self.m_g.get(key_g, 0)  # 如果键不存在，返回0
                    else:
                        key_f = 0
                        key_g = next((k for k in self.m_g.keys() if k.startswith(i[0]) and k.endswith(j[0])), None)
                        consumption_f = key_f  # 如果键不存在，返回0
                        consumption_g = self.m_g.get(key_g, 0)  # 如果键不存在，返回0
                # 为 q 的对应位置赋值

                q[i][j] = consumption_f + consumption_g
        for i in self.V_all:
            for j in self.V_all:
                if q[i][j] == 0:
                    q[i][j] = 100
        # 如果找不到路径则取100
        self.q = q
        
        self.model = None
        self.x = None
        self.e = None
        self.c = None

    def create_model(self):
        self.model = gp.Model("sub model")
        # 初始化子问题
        self.x = self.model.addVars(self.V_all, self.V_all, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='x')
        # 定义路径整数变量x
        self.e = self.model.addVars(self.V_all, lb=0, ub=GRB.INFINITY, name='e')
        # 定义电量变量e
        self.c = self.model.addVas(self.Se, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='c')
        # 定义在换电站i更换电池数量r
        self.u = self.model.addVars(self.V_all, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='c')

        for i in self.V_all:
            self.model.addConstr(gp.quicksum(self.x[i, j] for j in self.V_all) == 
                         gp.quicksum(self.x[j, i] for j in self.V_all), 
                         name=f"balance_{i}")
            self.model.addConstr(gp.quicksum(self.x[i, j] for j in self.V_all) <= 1, 
                         name=f"limit_{i}")
        # 流平衡约束 & 每个点最多被访问一次
        # 这里用的所有点入流出流相等，可能需要改成车库点以外
            
        self.model.addConstr(gp.quicksum( self.x[self.s][j] for j in self.V_all) == 1, name=f"balance_s")
        # 车库出流为1
        self.model.addConstr(gp.quicksum( self.x[j][self.s] for j in self.V_all) == 1,  name=f"balance_s2")
        # 车库入流为1

        self.model.addConstr(self.e[self.s] == self.Q,  name=f"energy_s")
        #车库出发时能量为满
        for i in self.Vs_all or self.s :
            for j in self.V_all :
                self.model.addConstr(self.e[i] - self.q[i][j] + M * (1 - self.x[i][j]) >= self.e[j],  name=f"energy_sVs_V")
        #从起点出发，到其他所有节点
        for i in self.Se :
            for j in self.V_all :
                self.model.addConstr(self.Q - self.q[i][j] + M * (1 - self.x[i][j]) >= self.e[j],  name=f"energy_Se_V")
        #从换电站到其他所有节点
                
        for i in self.Se :
            self.model.addConstr(self.c[i] * self.Q_0 >= self.Q - self.e[i],  name=f"ex_battery")
        #换电消耗电池数量
        
        # mtz消除子回路
        for i in self.V_all:
            for j in self.V_all:
                if i != j and i != self.s and j != self.s:  # 假设 s 是起始点
                    self.model.addConstr(self.u[i] - self.u[j] + len(self.V_all) * self.x[i, j] <= len(self.V_all) - 1)

        self.model.addConstr(gp.quicksum( self.x[i][j] for i in self.V_all for j in self.Se) <= 1,  name=f"ex_limit")
        #换电次数小于等于1
 
    def set_objective(self, q, mu, pi, sigma):
        self.model.setObjective(gp.quicksum(q[i][j] * self.x[i][j] for i in self.V_all for j in self.V_all)-gp.quicksum(mu * pi[i][j] * self.x[i][j] for i in self.V_all for j in self.V_all)-gp.quicksum(self.c[i] * sigma[i] for i in self.Sel), sense=GRB.MAXIMIZE)
        # 生成目的函数

    def solve(self, flag=0):
        self.model.Params.OutputFlag = flag  # 输出格式
        self.model.optimize()

    def get_solution(self):
        return [int(self.x[i].X) for i in self.V_all]
        # 获取主问题中x

    @property
    def get_obj(self):
        return self.model.ObjVal
        # 返回目标函数值

    def write(self):
        self.model.write("sub_model.lp")



# # 判断是否有子回路
# def has_subtour(graph, start_node):
#     def dfs(node, visited):
#         visited.add(node)
#         for neighbor in [n for n, connected in graph[node].items() if connected]:
#             if neighbor not in visited:
#                 dfs(neighbor, visited)

#     # 访问所有节点的深度优先搜索
#     visited = set()
#     dfs(start_node, visited)

#     # 检查是否所有非起始节点都被访问过
#     all_visited = all(node in visited for node in graph if node != start_node)
#     return not all_visited


# # 简化拓扑图
# def subtourck(my_x):
#     # 提取大于0的路径
#     active_paths = [item for item in my_x if my_x[item] > 0]
#     # 提取节点
#     nodes = sorted(set("".join(active_paths)))
#     # 创建一个初始为零的矩阵
#     graph_mat = np.zeros((len(nodes), len(nodes)))
#     # 为每个活跃路径设置值为1
#     for path in active_paths:
#         i, j = nodes.index(path[0]), nodes.index(path[1])
#         graph_mat[i, j] = 1
#     # 创建 DataFrame
#     graph_df = pd.DataFrame(graph_mat, index=nodes, columns=nodes, dtype=int)
#     return graph_df


# # callback
# def subtourelim(model, where):
#     if (where == GRB.Callback.MIPSOL):
#         # make a list of edges selected in the solution
#         solution_values = model.cbGetSolution(model._vars)
#         graph = subtourck(solution_values)
#         check = has_subtour(graph, "S")
#         if check:
#             print("---add sub tour elimination constraint--")

#             constraint_expr = gp.quicksum(model._vars[x] for x in solution_values if solution_values[x] > 0)
#             # add subtour elimination constraint 
#             model.cbLazy(constraint_expr <= sum(solution_values.values()) - 1)


if __name__ == "__main__":
    M = 50  # 总能耗上限
    order = pd.DataFrame([['A', 'B', 10], ['A', 'D', 8], ['B', 'C', 13], ['D', 'C', 4]],
                         columns=['start', 'end', 'weight'])
    location = pd.DataFrame([[0, 0], [1, 1], [4, 1], [1.5, -1], [5, -2]], index=['S', 'A', 'B', 'C', 'D'],
                            columns=['x', 'y'])
    pcost_f, pcost_g = 1, 0.8  # 单位距离满载/空载耗能
    _, V, F, m_f, G, m_g, L, h_gs, b_vl = data_pre(order, location, pcost_f, pcost_g)
    #  get from RMP RC
    pi = {'AB': 2, 'AD': 3, 'BC': 0.5, 'DC': 1}
    pi=Series(pi)
    sub_prob = SubProblem(pi, M, V, m_f, m_g, F, G, L, b_vl)
    sub_prob.create_model()
    sub_prob.set_objective(pi)
    sub_prob.solve()
    print(L)
    #  这里添加到主问题R中要注意 格式要改成一行 这里是一列
    print(sub_prob.get_solution())
    # 可视化生成的路线，可判断一下子回路判断及消除逻辑是否正确
    #     RI=pd.DataFrame([sub_prob.get_solution()],columns=L)
    #     route_visualise(0,RI,location,F,G)
    pass
