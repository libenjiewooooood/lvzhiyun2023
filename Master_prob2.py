import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from pandas import Series, DataFrame


class MasterProblem:
    def __init__(self, R: DataFrame, F: set or list, G: set or list, mu: float,
                 m_f: Series, m_g: Series, d_f: Series) -> None:
        # TODO 具体的输入格式，除路径集R以外，请参考data_process.py return中的描述
        self.R = R
        # R 全部的运输路径集合, 每一行代表一条运输路径
        """
        例如 R = 
           AB  DC  BA  BD  BS  CD  CA  CS  SA  SD
        0   9   0   9   0   1   0   0   0   1   0
        1   0   7   0   0   0   7   0   1   0   1
        2   5   2   4   1   0   1   0   1   1   0
        index=range(len(R)), name=L where L=F or G
        """
        self.F = F
        # 满载路段集合
        self.G = G
        # 空载路段集合
        self.d_f = d_f
        # 满载路段f的需求量

        self.mu = mu
        # mu 车辆最大载重量
        self.m_f = m_f
        self.m_g = m_g

        # m_f满载路径f耗电量
        # m_g空载路径g耗电量
        self.model = None
        self.y_r = None
        self.consts = None

    def create_model(self, relaxation=True):
        self.model = gp.Model("Master Problem")
        if relaxation:
            self.y_r = self.model.addVars(len(self.R), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='y_r')
        else:
            self.y_r = self.model.addVars(len(self.R), lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='y_r')
        # 定义整数变量y_r
        for arc in self.R.columns:
            if arc in self.F:
                col: pd.Series = self.R[arc]
                cont = gp.quicksum(self.y_r[i] * col[i] for i in range(len(self.R))) >= self.d_f[arc]/self.mu
                self.model.addConstr(cont, name=arc)
        # 订单需求约束

    def set_objective(self):
        self.model.setObjective(gp.quicksum(self.y_r[i] for i in range(len(self.R))), sense=GRB.MINIMIZE)
        # 目标函数

    def solve(self, flag=0):
        self.model.Params.OutputFlag = flag  # 输出格式
        self.model.optimize()  # 求解方法

    def get_dual_vars(self) -> Series:
        self.consts = self.model.getConstrs()
        pi = [con.Pi for con in self.consts]
        ct_name = [con.ConstrName for con in self.consts]
        return Series(pi, index=ct_name)  # 提取对偶值

    @property
    def opt(self):
        return self.model.objVal

    @property
    def solution(self):
        return [int(self.y_r[i].X) for i in range(len(self.R))]

    # def write(self):
    #     self.model.write("Master Problem.lp")


if __name__ == "__main__":
    F = ['AB', 'CD']
    G = ['SA', 'SC', 'DS', 'BS', 'DA', 'DC', 'BA', 'BC']
    R = DataFrame([[9, 0, 9, 0, 1, 0, 0, 0, 1, 0],
                   [0, 7, 0, 0, 0, 7, 0, 1, 0, 1],
                   [5, 2, 4, 1, 0, 1, 0, 1, 1, 0]],
                  index=range(3),
                  columns=F+G)
    mu = 3
    mf = Series([6, 7.28], index=F)
    mg = Series([1.69, 2.16, 6.46, 4.94, 6.00, 4.36, 33.6, 3.84], index=G)
    df = Series([20, 24], index=F)

    mp = MasterProblem(R=R, F=F, G=G, mu=mu, m_f=mf, m_g=mg, d_f=df)
    mp.create_model()
    mp.set_objective()
    mp.solve()
    pi = mp.get_dual_vars()
