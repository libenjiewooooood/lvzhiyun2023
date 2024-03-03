import pandas as pd
import matplotlib.pyplot as plt
from data_process import data_pre
from Subproblem import SubProblem
from Master_prob2 import MasterProblem
from visualise import order_visualise, route_visualise


def sub_route2master_route(sub_route: list, route_map: dict, L: list):
    data_frame_route = pd.Series(0, index=L)
    route_begin = [r[0] for r in sub_route]
    for i in range(len(route_begin) - 1):
        if route_begin[i] in route_map.keys() and route_begin[i + 1] in route_map.keys():
            data_frame_route[route_map[route_begin[i]]] += 1
            data_frame_route[route_map[route_begin[i]][::-1]] += 1
        elif route_begin[i] in route_map.keys() and route_begin[i + 1] not in route_map.keys():
            data_frame_route[route_map[route_begin[i]]] += 1
            str_route = route_map[route_begin[i]][-1] + route_begin[i + 1]
            data_frame_route[str_route] += 1
        else:
            str_route = route_begin[i] + route_begin[i + 1]
            data_frame_route[str_route] += 1
    return data_frame_route

# 输入数据
u = 3  # 货车最大载货量
order = pd.DataFrame([['A', 'B', 10], ['C', 'D', 13]],
                     columns=['start', 'end', 'weight'])
location = pd.DataFrame([[-2, 0], [2, 1], [1, 1], [4, 1], [1.5, -1], [5, -2], [0, 1], [2, 0]],
                        index=['S', 's', 'A', 'B', 'C', 'D', 'E', 'F'], columns=['x', 'y'])
pcost_f, pcost_g = 1, 0.8  # 单位距离满载/空载耗能
# 换电站集合
Se = {'E', 'F'}  # 充电站
S = {'S', 's'}  # 车库
df, V, Vs, F, m_f, G, m_g, L = data_pre(order, location, pcost_f, pcost_g, S, Se)
Q = 20  # 最大电池容量
Q_0 = 5  # 单节电池容量
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

# %% 主循环
iter_num = 0
while True:
    quit_loop = False
    # 求解主问题
    mp = MasterProblem(R, routes_power_consumption, c_kr, c_k, F, u, df)
    mp.create_model()
    mp.set_objective()
    mp.solve()
    sol = mp.solution
    pi = mp.get_order_dual_vars()
    sigma = mp.get_charge_dual_vars()
    print(f"--------Iter num:{iter_num}--------")
    # 求解子问题
    s = {'S'}  # 选中的车库
    sub_prob = SubProblem(pi, Vs, m_f, m_g, s, Se, Q, Q_0, u, sigma)
    sub_prob.create_model()
    sub_prob.set_objective()
    sub_prob.solve()
    severed_order, used_charge, sub_route_cost, new_route = sub_prob.get_solution()
    RC = (sub_route_cost - u * sum([severed_order[vs] * pi[vs] for vs in severed_order.keys()]) -
          sum([used_charge[ve] * sigma[ve] for ve in used_charge.keys()]))
    new_sub_route = sub_route2master_route(new_route, dict(zip(Vs, F)), L)
    if RC > 0:
        quit_loop = True
    else:
        if new_sub_route.tolist() not in R.to_numpy().tolist():
            print("Add a new route:")
            print(pd.DataFrame(new_sub_route).T)
            routes_power_consumption.append(sub_route_cost)
            c_kr.loc[len(c_kr)] = used_charge
            R.loc[len(R.index)] = new_sub_route
        else:
            print("Route repeat.")
            quit_loop = True
    iter_num += 1

    print('RELAXATION SOLUTION:')
    print(f"OPT={mp.opt()}")
    print(f"SOLUTION={sol}")
    solution_info = {i: v for i, v in enumerate(sol) if sol[i] > 0}
    print("Route Info:")
    print(R.iloc[list(solution_info.keys()), :])
    print("Solution Info:")
    print(pd.Series(solution_info))

    if quit_loop:
        print("--------SOLVE DONE.--------")
        mp = MasterProblem(R, routes_power_consumption, c_kr, c_k, F, u, df)
        mp.create_model(relaxation=False)
        mp.set_objective()
        mp.solve()
        print(f"OPT={mp.opt()}")
        print(f"SOLUTION={sol}")
        solution_info = {i: v for i, v in enumerate(sol) if sol[i] > 0}
        print("Route Info:")
        print(R.iloc[list(solution_info.keys()), :])
        print("Solution Info:")
        print(pd.Series(solution_info))
        print("Order Info:")
        print(order)
        # 最终解可视化
        # for i in solution_info.keys():
        #     route_visualise(i, R, location, F, G)
        # plt.show()
        break


