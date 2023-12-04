import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pandas import Series, DataFrame


class MasterProblem:
    def __init__(self, R: DataFrame, F: set or list, G: set or list, mu: float,
                 m_f: Series, m_g: Series, d_f: Series, V: set or list) -> None:
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
        self.V = V
        # V: 运输节点集合

        self.mu = mu
        # mu 车辆最大载重量
        self.m_f = m_f
        self.m_g = m_g

        # m_f满载路径f耗电量
        # m_g空载路径g耗电量
        self.model = None
        self.y_r = None
        self.consts = list()  # 约束集合

    def create_model(self):
        self.model = gp.Model("Master Problem")
        # TODO 根据路径集的大小确定决策变量的个数
        self.y_r = self.model.addVars(len(R), lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='y_r')
        # 定义整数变量y_r
        # TODO 不需要定义 a_fr, a_gr
        # self.a_fr = self.model.addVars(self.F, self.R, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='a_fr')
        # # 定义整数变量a_fr
        # self.a_gr = self.model.addVars(self.G, self.R, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='a_gr')
        # 定义整数变量a_gr
        for r in range(R.shape[1]):
            if R.columns=="AB"or"DC":
                elf.model.addConstr(
                    gp.quicksum(self.y_r[r] * R[f,r]) >= self.d_f[f]/mu for f in self.F)
        # TODO 约束形式为：对R中属于F的列，例如AB, DC列，点乘y_r, 大于等于 d_f/mu
        # TODO 将约束加入self.consts list中，并增加约束name，方便返回dual
        # self.model.addConstr(gp.quicksum(self.a_fr[f, r] * self.y_r[r] for r in self.R) >= self.d_f[f] for f in self.F)
        # self.model.addConstr(self.y_r[r] >= 0 for r in self.R)  # TODO 在定义决策变量的时候已经定义的大于0, 不需要增加约束
        # 订单需求约束

    def set_objective(self):
        self.model.setObjective(gp.quicksum(self.y_r[r] for r in self.R), sense=GRB.MINIMIZE)
        # 目标函数

    def solve(self, flag=0):
        self.model.Params.OutputFlag = flag  # 输出格式
        self.model.optimize()  # 求解方法

    def get_dual_vars(self) -> Series:
        return [self.consts[i].getAttr(GRB.Attr.Pi) for i in range(len(self.consts))]  # 提取对偶值

    # TODO 返回的对偶值，形式为Series, index=F

    def write(self):
        self.model.write("Master Problem.lp")


if __name__ == "__main__":
    R = DataFrame([[9, 0, 9, 0, 1, 0, 0, 0, 1, 0],
                   [0, 7, 0, 0, 0, 7, 0, 1, 0, 1],
                   [5, 2, 4, 1, 0, 1, 0, 1, 1, 0]],
                  index=range(3),
                  columns=["AB", "DC", "BA", "BD", "BS", "CD", "CA", "CS", "SA", "SD"])
    F = ['AB', 'AD', 'BC', 'DC']
    # F = ['AB']
    mu = 3
    print(R.shape[0])
    model = gp.Model("Master Problem")
    y_r = model.addVars(len(R), lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='y_r')
    # for f in range(len(F)):
    #     for r in range(R.shape[1]):
    #         for k in range(R.shape[0]):
    #             if F[f] in R.columns.tolist():
    for r in range(len(R)):
        for f in range(len(F)):
            for k in range(R.shape[1]):
                if F[f] in R.columns.tolist():
                    model.addConstrs(gp.quicksum(y_r[r] * R.loc[r, k]) >= (d_f[f]/mu))
    model.setObjective(gp.quicksum(y_r[r] for r in range(len(R))), sense=GRB.MINIMIZE)
    model.optimize()  # 求解
