import numpy as np
import gurobipy as gp
from gurobipy import GRB

class SubProblem:
    def __init__(self, pi, M, V, m_f, m_g, F, G , b, S, L) -> None:

        #############注意这里的pi,L和生成的x位置需要对应

        self.F = F 
        self.G = G
        #F 满载路段 G空载路段集合
        self.L = L
        #所有路段集合
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
        self.S = S
        #起点集合

    def create_model(self):
        self.model = gp.Model("sub model")
        #初始化子问题
        self.x = self.model.addVars(self.L, lb = 0, ub = GRB.INFINITY, vtype = GRB.INTEGER, name = 'x')
        #定义整数变量x
        self.model.addConstr(gp.quicksum(self.m_f[self.F.index(f)]*self.x[f] for f in self.F)+gp.quicksum(self.m_g[self.F.index(g)]*self.x[g] for g in self.G) <= self.M)
        #电池容量约束
        for v in self.V:
            self.model.addConstr(gp.quicksum(self.b[v,f]*self.x[f] for f in self.F) + gp.quicksum(self.b[v, g]*self.x[g] for g in self.G) == 0)
        #节点流平衡约束
        self.model.addConstr((gp.quicksum(self.b.loc['S',l].abs()*self.x[l] for l in self.L) == 2))
        #出发节点只有两条连线

    def set_objective(self, pi):
        self.model.setObjective(gp.quicksum(pi[self.x.index[f]]*self.x[f] for f in self.F), sense = GRB.MAXIMIZE)
        #生成目的函数    
        

    def solve(self, flag = 0):
        self.model.Params.OutputFlag = flag #输出格式
        self.model.optimize() #求解方法
    
    def get_solution(self):
        return [self.model.getVars()[i].x for i in self.L]
        #获取主问题中x

    def get_reduced_cost(self):
        return self.model.ObjVal
        #返回目标函数值
    
    def write(self):
        self.model.write("sub_model.lp")

