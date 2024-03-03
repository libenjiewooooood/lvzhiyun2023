import gurobipy as gp
from gurobipy import GRB
from pandas import Series, DataFrame

from data_process import data_pre


class MasterProblem:
    def __init__(self, R: DataFrame, er: list, c_kr: DataFrame, c_k: Series, F: set or list, mu: float,
                 d_f: Series) -> None:
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
        assert R.shape[0] == len(er)
        self.er = er
        # 表示每条路径的电量消耗
        self.c_kr = c_kr
        # 表示路径r在换电站k更换的电池数量
        """
        例如 c_kr = 
            E   F
        0   0   1
        1   1   0
        2   0   2
        表示路径0在换电站F更换了1块电池；路径1在换电站E更换了1块电池;路径2在换电站F更换了2块电池
        """
        assert len(c_k) == c_kr.shape[1]
        self.c_k = c_k  # 例如c_k = Series([10, 10], index=['E', 'F'])
        # 表示换电站k的最大可用电池数量
        self.d_f = d_f
        # 满载路段f的需求量

        self.mu = mu
        # mu 车辆最大载重量

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
                cont = gp.quicksum(self.y_r[i] * col[i] for i in range(len(self.R))) >= self.d_f[arc] / self.mu
                self.model.addConstr(cont, name=arc)
        # 订单需求约束
        for k in self.c_kr.columns:
            col: pd.Series = self.c_kr[k]
            cont = gp.quicksum(self.y_r[i] * col[i] for i in range(len(self.c_kr))) <= self.c_k[k] / self.mu
            self.model.addConstr(cont, name=k)

    def set_objective(self):
        self.model.setObjective(gp.quicksum(self.y_r[i] * self.er[i] for i in range(len(self.R))), sense=GRB.MINIMIZE)
        # 目标函数

    def solve(self, flag=0):
        self.model.Params.OutputFlag = flag  # 输出格式
        self.model.optimize()
        # print("Model Status:", self.model.Status)
        if self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")

    def get_order_dual_vars(self) -> dict:
        dual_values_arc = {}
        for arc in self.R.columns:
            if arc in self.F:
                dual_values_arc[arc[0]] = self.model.getConstrByName(arc).Pi
        return dual_values_arc

    def get_charge_dual_vars(self) -> dict:
        dual_values_k = {}
        for k in self.c_kr.columns:
            dual_values_k[k] = self.model.getConstrByName(k).Pi
        return dual_values_k

    def opt(self):
        if self.model.status == GRB.OPTIMAL:
            return self.model.objVal
        else:
            raise Exception('Not solved')

    @property
    def solution(self):
        return [float(self.y_r[i].X) for i in range(len(self.R))]

    # def write(self):
    #     self.model.write("Master Problem.lp")


if __name__ == "__main__":
    u = 3  # 货车最大载货量
    order = DataFrame([['A', 'B', 10], ['C', 'D', 13]],
                         columns=['start', 'end', 'weight'])
    location = DataFrame([[-2, 0], [2, 1], [1, 1], [4, 1], [1.5, -1], [5, -2], [0, 1], [2, 0]],
                            index=['S', 's', 'A', 'B', 'C', 'D', 'E', 'F'], columns=['x', 'y'])
    pcost_f, pcost_g = 1, 0.8  # 单位距离满载/空载耗能
    # 换电站集合
    Se = {'E', 'F'}  # 充电站
    S = {'S', 's'}  # 车库
    df, V, Vs, F, m_f, G, m_g, L = data_pre(order, location, pcost_f, pcost_g, S, Se)

    # %开始求解
    # 初始化解
    R = DataFrame([[0] * len(L) for _ in range(len(F))], columns=L)  # 为每一个订单初始化一条运输路径
    c_kr = DataFrame(0, columns=list(Se), index=range(len(order)))  # 每条路径使用电池情况
    c_k = Series([10, 10], index=Se)  # 换电站最大电池数量

    routes_power_consumption = []
    for i in range(len(order)):
        laden_sect: str = F[i]
        idle_sect_0 = 'S' + laden_sect[0]
        idle_sect_1 = laden_sect[1] + 'S'
        R.loc[i, [laden_sect, idle_sect_0, idle_sect_1]] = 1
        power_consumption = m_f[laden_sect] + m_g[idle_sect_0] + m_g[idle_sect_1]
        routes_power_consumption.append(power_consumption)

    mp = MasterProblem(R, routes_power_consumption, c_kr, c_k, F, u, df)
    mp.create_model()
    mp.set_objective()
    mp.solve()
    print(mp.solution)
    print(mp.opt())
    print(mp.get_order_dual_vars())
    print(mp.get_charge_dual_vars())
