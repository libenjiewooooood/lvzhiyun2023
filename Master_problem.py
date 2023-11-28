import numpy as np
import gurobipy as gp
from gurobipy import GRB

class Master:
    def __init__(self, R, F, G, M,U,m_f: list, m_g: list, d_f: list,S: list, E: list) -> None:
        self.R = R
        # R 全部的运输路径集合
        self.F = F
        # 满载路段集合
        self.G = G
        # 空载路段集合
        self.d_f = d_f
        # 满载路段f的需求量
        self.S = S
        self.E = E
        # S 出发点节点集合，E 目的地节点集合
        self.M = M
        # M 电池最大容量
        self.U = U
        # U 车辆最大载重量
        self.m_f = m_f
        self.m_g = m_g

        # m_f满载路径f耗电量
        # m_g空载路径g耗电量

    def create_model(self):
        self.model = gp.Model("Master Problem")
        self.y_r = self.model.addVars(self.R, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='y_r')
        # 定义整数变量y_r
        self.a_fr = self.model.addVars(self.F,self.R, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='a_fr')
        # 定义整数变量a_fr
        self.a_gr = self.model.addVars(self.G,self.R, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='a_gr')
        # 定义整数变量a_gr
        self.model.addConstr(gp.quicksum(self.a_fr[f,r] * self.y_r[r] for r in self.R) >= self.d_f[f] for f in self.F)
        self.model.addConstr(self.y_r[r] >= 0 for r in self.R)
        # 订单需求约束
    def set_objective(self):
        self.model.setObjective(gp.quicksum(self.y_r[r] for r in self.R), sense = GRB.MINIMIZE)
        # 目标函数
    def solve(self, flag = 0):
        self.model.Params.OutputFlag = flag #输出格式
        self.model.optimize() #求解方法
    def get_dual_vars(self):
        return [self.constrs[i].getAttr(GRB.Attr.Pi) for i in range(len(self.constrs))] #提取对偶值
    def write(self):
        self.model.write("Master Problem.lp")


