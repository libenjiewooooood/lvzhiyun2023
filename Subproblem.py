import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from pandas import Series, DataFrame



class SubProblem:
    def __init__(self, pi: Series, M: float, V: set or list[str], m_f: Series, m_g: Series,
                 F: set or list[str], G: set or list[str], L: set or list[str], b_vl: DataFrame) -> None:
        #############注意这里的pi,L和生成的x位置需要对应

        self.F = F  # 例如['AB', 'DC']
        self.G = G  # 例如['SA', 'SD', 'CS', 'BS', 'CA', 'CD', 'BA', 'BD']
        # F 满载路段 G空载路段集合
        self.L = L
        self.V = V  # 例如['A', 'B', 'C', 'D', 'S']
        # V 节点集合
        # 所有路段集合
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
        battery_ct = gp.quicksum(self.m_f[self.F.index(f)] * self.x[f] for f in self.F) + \
                     gp.quicksum(self.m_g[self.G.index(g)] * self.x[g] for g in self.G) <= self.M
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
        self.model.Params.OutputFlag = flag  # 输出格式
        self.model.optimize()  # 求解方法

    def get_solution(self):
        return [self.x[l].X for l in self.L]
        # 获取主问题中x

    def get_reduced_cost(self):
        return self.model.ObjVal
        # 返回目标函数值

    def write(self):
        self.model.write("sub_model.lp")


if __name__ == "__main__":
    M = 50  # 总能耗上限
    V = ['A', 'C', 'D', 'B', 'S']
    F = ['AB', 'AD', 'BC', 'DC']
    m_f = [3.0, 5.0, 3.2015621187164243, 3.640054944640259]
    G = ['SA', 'SB', 'SD', 'BS', 'CS', 'DS', 'BA', 'BD', 'CA', 'CB', 'CD', 'DA', 'DB']
    m_g = [1.1313708498984762, 3.2984845004941286, 4.308131845707603, 3.2984845004941286, 1.4422205101855958, 4.308131845707603, 2.4000000000000004, 2.529822128134704, 1.6492422502470643, 2.5612496949731396, 2.9120439557122073, 4.0, 2.529822128134704]
    L = F + G  # 假设 F 和 G 之间没有重复元素
    # 流矩阵数据
    b_vl_data = {
        'AB': [-1, 0, 0, 1, 0],
        'AD': [-1, 0, 1, 0, 0],
        'BC': [0, 1, 0, -1, 0],
        'DC': [0, 1, -1, 0, 0],
        'SA': [1, 0, 0, 0, -1],
        'SB': [0, 0, 0, 1, -1],
        'SD': [0, 0, 1, 0, -1],
        'BS': [0, 0, 0, -1, 1],
        'CS': [0, -1, 0, 0, 1],
        'DS': [0, 0, -1, 0, 1],
        'BA': [1, 0, 0, -1, 0],
        'BD': [0, 0, 1, -1, 0],
        'CA': [1, -1, 0, 0, 0],
        'CB': [0, -1, 0, 1, 0],
        'CD': [0, -1, 1, 0, 0],
        'DA': [1, 0, -1, 0, 0],
        'DB': [0, 0, -1, 1, 0]
    }
    b_vl = pd.DataFrame(b_vl_data, index=V)


    pi={'AB': 2, 'AD': 0, 'BC': 1, 'DC': 0}
    sub_prob = SubProblem(pi, M, V, m_f, m_g, F, G, L, b_vl)
    sub_prob.create_model()
    sub_prob.set_objective(pi)
    sub_prob.solve()
    print(sub_prob.get_solution())
    pass
