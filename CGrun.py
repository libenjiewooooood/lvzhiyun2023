import pandas as pd
import matplotlib.pyplot as plt
from data_process import data_pre
from Subproblem import SubProblem
from Master_prob2 import MasterProblem
from visualise import order_visualise, route_visualise

# 输入数据
u = 3  # 货车最大载货量
order = pd.DataFrame([['A', 'B', 10], ['A', 'D', 8], ['B', 'C', 13], ['D', 'C', 4]],
                     columns=['start', 'end', 'weight'])
location = pd.DataFrame([[-2, 0], [2, 1], [1, 1], [4, 1], [1.5, -1], [5, -2], [0, 1], [2, 0]],
                        index=['S', 's', 'A', 'B', 'C', 'D', 'E', 'F'], columns=['x', 'y'])
pcost_f, pcost_g = 1, 0.8  # 单位距离满载/空载耗能
# 换电站集合
Se = {'E', 'F'}  # 充电站
S = {'S', 's'}  # 车库
df, V, Vs, F, m_f, G, m_g, L = data_pre(order, location, pcost_f, pcost_g, S, Se)

# %开始求解
# 初始化解
R = pd.DataFrame([[0] * len(L) for _ in range(len(F))], columns=L)  # 为每一个订单初始化一条运输路径
c_kr = pd.DataFrame(0, columns=list(Se), index=range(len(order)))  # 每条路径使用电池情况
c_k = pd.Series([10, 10], index=Se)  # 换电站最大电池数量

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
#%% 主循环

iter_num = 1
while True:
    quit_loop = False
    # 求解主问题
    mp = MasterProblem(R, routes_power_consumption, c_kr, c_k, F, u, df)
    mp.create_model()
    mp.set_objective()
    mp.solve()
    pi = mp.get_order_dual_vars()
    gamma = mp.get_charge_dual_vars()
    print(f"--------Iter num:{iter_num}--------")
    # 求解子问题
    sub_prob = SubProblem(pi, m, V, m_f, m_g, F, G, L, b_vl)
    sub_prob.create_model()
    sub_prob.set_objective(pi)
    sub_prob.solve()
    # print(L)
    if sub_prob.get_obj * mu <= 1:
        quit_loop = True
    else:
        new_solution = sub_prob.get_solution()

        list_R = R.to_numpy().tolist()
        if new_route not in list_R:
            print("Add a new route:")
            print(pd.Series(new_route, index=L))
            R.loc[len(R.index)] = new_route
        else:
            print("Route repeat.")
            quit_loop = True
    iter_num += 1
    if quit_loop:
        mp = MasterProblem(R=R, F=F, G=G, mu=mu, m_f=m_f, m_g=m_g, d_f=df)
        mp.create_model(relaxation=False)
        mp.set_objective()
        mp.solve()
        sol = mp.solution
        print("SOLVE DONE.")
        print(f"OPT={mp.opt}")
        print(f"SOLUTION={sol}")
        solution_info = {i: v for i, v in enumerate(sol) if sol[i] > 0}
        print("Route Info:")
        print(R.iloc[list(solution_info.keys()), :])
        print("Solution Info:")
        print(pd.Series(solution_info))
        print("Order Info:")
        print(order)
        # 最终解可视化
        for i in solution_info.keys():
            route_visualise(i, R, location, F, G)
        plt.show()
        break

'''
