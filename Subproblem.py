import numpy as np
import gurobipy as gp
from gurobipy import GRB

class SubProblem:
    def __init__(self, pi:list, M, V, m_f: list, m_g: list, F : list, G :list, b, h) -> None:
        self.F = F 
        self.G = G
        #F 出发点节点集合，G目的地节点集合
        self.M = M 
         #M 电池最大容量
        self.m_f = m_f
        self.m_g = m_g
        #m_f满载路径f耗电量
        #m_g空载路径g耗电量
        self.V = V
        #V 节点集合
        self.b = b
        #流平衡矩阵
        self.h = h
        #终点返回起点消耗电量
        
        
    def create_model(self):
        self.model = gp.Model("sub model")
        #初始化子问题
        self.x = self.model.addVars(self.M, lb = 0, ub = GRB.INFINITY, vtype = GRB.INTEGER, name = 'x')
        #定义整数变量x
        self.model.addConstr((gp.quicksum(self.m_f[f]*self.x[f]+self.m_g*self.x[g] for f in range(self.F) for g in range(self.G) ) <= self.M))
        #电池容量约束
        for v in self.V:
            self.model.addConstr(gp.quicksum(self.b[v, f]*self.x[f] for f in self.F) + gp.quicksum(self.b[v, g]*self.x[g] for g in self.G) == 0)
        #节点流平衡约束
        self.model.addConstr((gp.quicksum(self.h[g]*self.x[g] for g in range(self.G) ) == 1))
        #回到出发点流为1

    def set_objective(self, pi):
        self.model.setObjective(gp.quicksum(pi[f]*self.x[f] for f in range(self.F)), sense = GRB.MAXIMIZE)
        #生成目的函数

    def solve(self, flag = 0):
        self.model.Params.OutputFlag = flag #输出格式
        self.model.optimize() #求解方法
    
    def get_solution(self):
        return [self.model.getVars()[i].x for i in range(self.M)]
        #获取主问题中x

    def get_reduced_cost(self):
        return self.model.ObjVal
        #返回目标函数值
    
    def write(self):
        self.model.write("sub_model.lp")

