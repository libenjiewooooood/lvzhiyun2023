import numpy as np
import gurobipy as gp
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
        battery_ct = gp.quicksum(self.m_f[f] * self.x[f] for f in self.F) + \
                     gp.quicksum(self.m_g[g] * self.x[g] for g in self.G) <= self.M
        self.model.addConstr(battery_ct, name="battery_ct")
        # 电池容量约束
        # TODO 对关联矩阵b_vl加约束
        for v in self.V:
            self.model.addConstr(gp.quicksum(self.b.loc[v, f] * self.x[f] for f in self.F) + gp.quicksum(
                self.b.loc[v, g] * self.x[g] for g in self.G) == 0, name=f'vector{v}')
        # 节点流平衡约束
        self.model.addConstr((gp.quicksum(self.b.loc['S', l].abs() * self.x[l] for l in self.L) == 2))
        # 出发节点只有两条连线

    def set_objective(self, pi):
        self.model.setObjective(gp.quicksum(pi[self.x.index[f]] * self.x[f] for f in self.F), sense=GRB.MAXIMIZE)
        # 生成目的函数

    def solve(self, flag=0):
        self.model.Params.OutputFlag = flag  # 输出格式
        self.model.optimize()  # 求解方法

    def get_solution(self):
        return [self.model.getVars()[i].x for i in self.L]
        # 获取主问题中x

    def get_reduced_cost(self):
        return self.model.ObjVal
        # 返回目标函数值

    def write(self):
        self.model.write("sub_model.lp")


if __name__ == "__main__":
    pass
