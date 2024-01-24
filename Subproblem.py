import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from pandas import Series, DataFrame
from data_process import data_pre
# from visualise import route_visualise
# from data_process import data_pre


class SubProblem:
    def __init__(self, pi: Series, Vs: set or list[str], m_f: Series, m_g: Series,
                 s: set or list[str], Se: set or list[str], Q, Q_0, mu, sigma) -> None:
        self.Vs = Vs #订单起点集合
        self.Vs_all = [f"{node}{i}" for node in Vs for i in range(1, 4)] #复制后的订单起点
        # print(self.Vs_all)
        self.pi = pi #RMP问题得到的执行满载路段的对偶值
        pi_all = {}  # 扩展后的 pi 字典
        # 遍历原始 pi 字典中的每个键和值
        for node, value in pi.items():
            # 为每个原始节点创建扩展键
            for i in range(1, 4):
                pi_key = f"{node}{i}"
                pi_all[pi_key] = value
        self.pi_all = pd.Series(pi_all) #复制后的RMP问题对偶值
        # print('pi_all')
        # print(self.pi_all)
        # self.Ve = Ve #订单终点集合
        self.s = s #指定车库为起点
        self.Se = Se #换电站集合
        self.V_all = list(set(self.Vs_all)|s|Se)  # 所有需要节点集合例如A1,A2,A3,C1,C2,C3,S,s,E,F
        self.Q = Q  # 最大电池容量
        self.Q_0 = Q_0  # 换电站更换的单节电池容量
        M = 1000 # M一个极大的数
        self.m_f = m_f # m_f满载路径f耗电量
        self.m_g = m_g # m_g空载路径g耗电量
        #生成两点之间的耗电量
        q = pd.DataFrame(0.0, index=self.V_all, columns=self.V_all)
        #print(q)
        for i in self.V_all:
            for j in self.V_all:
                if i != j:
                    if i[0] in Vs:
                    # 为 m_f 找到以 i[0] 开头的键
                        key_f = next((k for k in m_f.keys() if k.startswith(i[0])), None)

                        # 为 m_g 找到以 j 结尾的键
                        key_g = next((k for k in m_g.keys() if k.startswith(key_f[1]) and k.endswith(j[0])), None)

                        # 从 m_f 和 m_g 中获取对应的电量消耗值
                        consumption_f = m_f.get(key_f, 0)  # 如果键不存在，返回0
                        consumption_g = m_g.get(key_g, 0)  # 如果键不存在，返回0
                    else:
                        key_f = 0
                        key_g = next((k for k in m_g.keys() if k.startswith(i[0]) and k.endswith(j[0])), None)
                        consumption_f = key_f  # 如果键不存在，返回0
                        consumption_g = m_g.get(key_g, 0)  # 如果键不存在，返回0
                    # 为 q 的对应位置赋值
                    q[i][j] = consumption_f + consumption_g
                    # if i == 'F' and j == 'A1':
                    #     print('q(F,A1)')
                    #     print(i[0],j[0])
                    #     print(q[i][j])
        for i in self.V_all:
            for j in self.V_all:
                if q[i][j] == 0:
                    q[i][j] = 101
        # 如果找不到路径则取99
        self.q = q
        # print(self.q)
        # print(self.q['A1']['A1'])
        # print(self.q['A1']['A3'])
        # print('self.q.at[A1, C1]')
        # print(self.q.at['A1', 'C1'])
        self.model = None
        self.x = None
        self.e = None
        self.c = None
        # print('self.Vs_all')
        # print(self.Vs_all)

        

    def create_model(self):
        self.model = gp.Model("sub model")
        # 初始化子问题
        self.x = self.model.addVars(self.V_all, self.V_all, lb=0, ub=1, vtype=GRB.INTEGER, name='x')
        # 定义路径0-1变量x
        self.e = self.model.addVars(self.V_all, lb=0, ub=self.Q, name='e')
        # 定义电量变量e
        self.c = self.model.addVars(self.Se, lb=0, ub=10 , vtype=GRB.INTEGER, name='c')
        # 定义在换电站i更换电池数量r
        self.u = self.model.addVars(self.V_all, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='u')
        # u表示访问顺序，用于除去子回路

        for i in self.V_all:
            self.model.addConstr(gp.quicksum(self.x[i, j] for j in self.V_all) == 
                         gp.quicksum(self.x[j, i] for j in self.V_all), 
                         name=f"balance_{i}")
            self.model.addConstr(gp.quicksum(self.x[i, j] for j in self.V_all) <= 1, 
                         name=f"limit_{i}")
        # 流平衡约束 & 每个点最多被访问一次
        
        for i in self.s:
            self.model.addConstr(gp.quicksum( self.x[i,j] for j in self.V_all) == 1, name=f"balance_s")
            # 车库出度为1
            self.model.addConstr(gp.quicksum( self.x[j,i] for j in self.V_all) == 1,  name=f"balance_s2")
            # 车库入度为1

        for i in self.s:
            self.model.addConstr(self.e[i] == self.Q,  name=f"energy_s")
        #车库出发时能量为满
        
        for i in self.Vs_all :
            for j in self.Vs_all:
                if i != j:
                    self.model.addConstr(self.e[i] - self.q.at[i, j] + M * (1 - self.x[i,j]) >= self.e[j],  name=f"energy_1_1")
        #从起点出发，到起点的电量约束
                
        for i in self.Vs_all:
            for j in self.Se:
                self.model.addConstr(self.e[i] - self.q.at[i, j] + M * (1 - self.x[i,j]) >= self.e[j],  name=f"energy_1_2")
        #从起点出发，到充电桩的电量约束
                
        for i in self.s :
            for j in self.Vs_all:
                self.model.addConstr(self.Q - self.q.at[i, j] + M * (1 - self.x[i,j]) >= self.e[j],  name=f"energy_1_3")
        #从车库出发，到起点的电量约束
                
                                
        for i in self.Se:
            for j in self.Vs_all:
                self.model.addConstr(self.Q - self.q.at[i, j] + M * (1 - self.x[i,j]) >= self.e[j],  name=f"energy_2")
        #从换电站到其他所有节点的电量约束
                
        for i in self.Vs_all:
            for j in self.s:
                self.model.addConstr(self.e[i] - self.q.at[i, j] + M * (1 - self.x[i,j]) >= 0,  name=f"energy_3")
        #从起点回到车库的电量约束
                
        for i in self.Se:
            for j in self.s:
                self.model.addConstr(self.Q - self.q.at[i, j] + M * (1 - self.x[i,j]) >= 0,  name=f"energy_4")
        #从充电桩回到车库的电量约束
                
        for i in self.Se :
            self.model.addConstr(self.c[i] * self.Q_0 >= self.Q - self.e[i],  name=f"ex_battery")
        #换电消耗电池数量
        
        # mtz消除子回路
        for i in self.Vs_all or self.Se:
            for j in self.Vs_all or self.Se:
                # if i != j and i != self.s and j != self.s:  # 假设 s 是起始点
                self.model.addConstr(self.u[i] - self.u[j] + len(self.V_all) * self.x[i, j] <= len(self.V_all) - 1,  name=f"mtz")


        self.model.addConstr(gp.quicksum( self.x[i,j] for i in self.V_all for j in self.Se) <= 1,  name=f"ex_limit")
        #换电次数小于等于1
 
    def set_objective(self, mu, sigma):
        # print("V_all:", self.V_all)
        # print("q indices:", self.q.index, self.q.columns)
        # print("q:", self.q)
        # print('self.pi_all:', self.pi_all )
        self.model.setObjective(gp.quicksum(self.q.at[i, j] * self.x[i,j] for i in self.V_all for j in self.V_all)-gp.quicksum(mu * self.pi_all[i] * gp.quicksum(self.x[i,j] for j in self.V_all) for i in self.Vs_all)-gp.quicksum(self.c[i] * sigma[i] for i in self.Se), sense=GRB.MINIMIZE)
        # 生成目的函数

    def solve(self, flag=0):
        self.model.Params.OutputFlag = flag  # 输出格式
        self.model.optimize()
        print("Model Status:", self.model.Status)
        if self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")

    def get_solution(self):
        solutionx = [int(self.x[i,j].X) for i in self.V_all for j in self.V_all]
        solutione = [int(self.e[i].X) for i in self.V_all]
        solutionc = [int(self.c[i].X) for i in self.Se]
        solutionu = [int(self.u[i].X) for i in self.V_all]
        # print(self.Se)
        print(self.V_all)
        return solutionx, solutione, solutionc, solutionu

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
    order = pd.DataFrame([['A', 'B', 10], ['C', 'D', 13]],
                         columns=['start', 'end', 'weight'])
    location = pd.DataFrame([[-2, 0], [2,1],[1, 1], [4, 1], [1.5, -1], [5, -2], [0, 1], [2, 0]],
                            index=['S','s', 'A', 'B', 'C', 'D', 'E', 'F'], columns=['x', 'y'])
    pcost_f, pcost_g = 1, 0.8  # 单位距离满载/空载耗能
    # 换电站集合
    Se = {'E','F'}
    # 车库集合
    S = {'S','s'}
    df,V,Vs,F,m_f,G,m_g,L=data_pre(order,location,pcost_f,pcost_g, S, Se)
    #  get from RMP RC
    pi = {'A': 5, 'C': 5}
    pi = Series(pi)
    # RMP问题满载路径的对偶值
    Q = 100 #最大电池容量
    Q_0 = 20 #单节电池容量
    mu = 3 # 最大载重
    sigma = {'E': 2, 'F': 2}
    sigma = Series(sigma)
    s = {'S'}# 选中的车库
    sub_prob = SubProblem(pi, Vs, m_f, m_g, s, Se, Q, Q_0, mu, sigma)
    sub_prob.create_model()
    sub_prob.set_objective(mu, sigma)
    sub_prob.solve()
    print(sub_prob.get_solution())
    # 可视化生成的路线，可判断一下子回路判断及消除逻辑是否正确
    #     RI=pd.DataFrame([sub_prob.get_solution()],columns=L)
    #     route_visualise(0,RI,location,F,G)
    pass
