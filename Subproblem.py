import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from pandas import Series, DataFrame



class SubProblem:
    def __init__(self, pi: Series, M: float, V: set or list[str], m_f: Series, m_g: Series,
                 F: set or list[str], G: set or list[str], L: set or list[str], b_vl: DataFrame) -> None:
        self.F = F  # 例如['AB', 'DC']
        self.G = G  # 例如['SA', 'SD', 'CS', 'BS', 'CA', 'CD', 'BA', 'BD']
        # F 满载路段 G空载路段集合
        self.L = L
        # 所有路段集合
        self.V = V  # 例如['A', 'B', 'C', 'D', 'S']
        # V 节点集合
        self.M = M
        # M 电池最大容量
        self.m_f = m_f
        self.m_g = m_g
        # m_f满载路径f耗电量
        # m_g空载路径g耗电量
        self.b_vl = b_vl  # 关联矩阵
        """
        b_vl =
           AB  DC  SA  SD  CS  BS  CA  CD  BA  BD
        S   0   0  -1  -1   1   1   0   0   0   0
        B   1   0   0   0   0  -1   0   0  -1  -1
        A  -1   0   1   0   0   0   1   0   1   0
        C   0   1   0   0  -1   0  -1  -1   0   0
        D   0  -1   0   1   0   0   0   1   0   1
        index=S, name=L
        """
        # 流平衡矩阵，必为M, 0或者1，不需要额外定义

        self.model = None
        self.x = None

    def create_model(self):
        self.model = gp.Model("sub model")
        # 初始化子问题
        self.x = self.model.addVars(self.L, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='x')
        # 定义整数变量x
        battery_ct = gp.quicksum(self.m_f[f] * self.x[f] for f in self.F) + \
                     gp.quicksum(self.m_g[g] * self.x[g] for g in self.G) <= self.M
        self.model.addConstr(battery_ct, name="battery_ct")
        # 电池容量约束
        # TODO 对关联矩阵b_vl加约束
        for v in self.V:
            self.model.addConstr(gp.quicksum(self.b_vl.loc[v, f] * self.x[f] for f in self.F) + gp.quicksum(
                self.b_vl.loc[v, g] * self.x[g] for g in self.G) == 0, name=f'vector{v}')
        # 节点流平衡约束
        self.model.addConstr((gp.quicksum(abs(self.b_vl.loc['S', l]) * self.x[l] for l in self.L) == 2))
        # 出发节点只有两条连线

    def set_objective(self, pi):
        self.model.setObjective(gp.quicksum(pi[f] * self.x[f] for f in self.F), sense=GRB.MAXIMIZE)
        # 生成目的函数

    def solve(self, flag=0):
        self.model.Params.OutputFlag = flag
        # self.model.optimize()

        # graph = self.subtourck()
        # check = has_subtour(graph,"S")

        while True:
            self.model.optimize()

            # 检查模型是否成功优化
            if self.model.Status != gp.GRB.OPTIMAL:
                print("Optimization was not successful.")
                break

            graph = self.subtourck()
            check = has_subtour(graph,"S")

            if not check:
                break  # 如果没有子回路，退出循环

            solution_values = [self.x[l].X for l in self.L]
            solution_dict = dict(zip(self.L, solution_values))

            constraint_expr = gp.quicksum((1 if solution_dict[l] > 0 else 0) * self.x[l] for l in self.L)
            total_value = sum(solution_dict[l] for l in self.L) - 1
            self.model.addConstr(constraint_expr <= total_value)
            #有子回路则添加约束舍去这个路线
            self.model.update()  # 更新模型

        if self.model.Status == gp.GRB.OPTIMAL:
            print("Solution:", [self.x[l].X for l in self.L])
        else:
            print("No optimal solution found.")
            
    def get_solution(self):
        return [int(self.x[l].X) for l in self.L]
        # 获取主问题中x

    @property
    def get_obj(self):
        return self.model.ObjVal
        # 返回目标函数值

    def write(self):
        self.model.write("sub_model.lp")


    #生成路线图的简化版
    def subtourck(self):
        my_x = self.get_solution()
        # 提取大于0的路径
        active_paths = [l for l, x in zip(self.L, my_x) if x > 0]
        # 提取节点
        nodes = sorted(set("".join(active_paths)))
        # 创建一个初始为零的矩阵
        graph_mat = np.zeros((len(nodes), len(nodes)))
        # 为每个活跃路径设置值为1
        for path in active_paths:
            i, j = nodes.index(path[0]), nodes.index(path[1])
            graph_mat[i, j] = 1
        # 创建 DataFrame
        graph_df = pd.DataFrame(graph_mat, index=nodes, columns=nodes , dtype=int)
        return graph_df
    
    # def subtour_elimination(self, where):
    #     if where == gp.GRB.Callback.MIPSOL:
    #         # 使用 cbGetSolution 获取当前解决方案
    #         solution_values = self.model.GetSolution([self.x[l] for l in self.L])
    #         solution_dict = dict(zip(self.L, solution_values))

    #         # 检查是否有子回路
    #         graph = self.subtourck()  # 确保这个方法能正确生成图
    #         check = has_subtour(graph)  # 确保这个方法能正确检测子回路

    #         return check

    #     if check:
    #         # 如果有子回路，构建并添加 Lazy 约束
    #         constraint_expr = gp.quicksum(solution_dict[l] * self.x[l] for l in self.L)
    #         total_value = sum(solution_dict[l] for l in self.L) - 1
    #         self.model.cbLazy(constraint_expr <= total_value)

    # def solve(self, flag=0):
    #     self.model.Params.OutputFlag = flag
    #     self.model.Params.lazyConstraints = 1
    #     self.model.optimize(self.subtour_elimination)


#判断是否有子回路
def has_subtour(graph, start_node):
    def dfs(node, visited):
        visited.add(node)
        for neighbor in [n for n, connected in graph[node].items() if connected]:
            if neighbor not in visited:
                dfs(neighbor, visited)

    # 访问所有节点的深度优先搜索
    visited = set()
    dfs(start_node, visited)

    # 检查是否所有非起始节点都被访问过
    all_visited = all(node in visited for node in graph if node != start_node)
    return not all_visited

  




if __name__ == "__main__":
    M = 50  # 总能耗上限
    V = ['A', 'B', 'D', 'S', 'C']
    F = ['AB', 'CD']
    m_f = pd.Series({'AB': 6.00000, 'CD': 7.28011})
    G = ['SA', 'SC', 'DS', 'BS', 'DA', 'DC', 'BA', 'BC']
    m_g = pd.Series({'SA': 1.697056, 'SC': 2.163331, 'DS': 6.462198, 'BS': 4.947727, 
                'DA': 6.000000, 'DC': 4.368066, 'BA': 3.600000, 'BC': 3.841875})
    L = F + G  # 假设 F 和 G 之间没有重复元素
    #流矩阵数据
    b_vl_data = {
       'AB': [-1, 1, 0, 0, 0],
       'CD': [0, 0, 1, 0, -1],
       'SA': [1, 0, 0, -1, 0],
       'SC': [0, 0, 0, -1, 1],
       'DS': [0, 0, -1, 1, 0],
       'BS': [0, -1, 0, 1, 0],
       'DA': [1, 0, -1, 0, 0],
       'DC': [0, 0, -1, 0, 1],
       'BA': [1, -1, 0, 0, 0],
       'BC': [0, -1, 0, 0, 1]
    }
    b_vl = pd.DataFrame(b_vl_data, index=V)
    pi={'AB': 0.1, 'CD': 0.125}
    sub_prob = SubProblem(pi, M, V, m_f, m_g, F, G, L, b_vl)
    sub_prob.create_model()
    sub_prob.set_objective(pi)
    sub_prob.solve()
    print(L)
    print("solotion" ,sub_prob.get_solution())
    graph=sub_prob.subtourck()
    print(graph)
    print(has_subtour(graph,"S"))

    pass
