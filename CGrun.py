import pandas as pd

from data_process import data_pre
from Subproblem import SubProblem
from Master_prob2 import MasterProblem
from visualise import order_visualise, route_visualise

# 输入数据
mu = 3  # 货车最大载货量
m = 100  # 最大电容量s
# 订单
order = pd.DataFrame([['A', 'B', 20],
                      ['D', 'C', 30],
                      ['E', 'F', 40]], columns=['start', 'end', 'weight'])
# 货运节点
location = pd.DataFrame([[0, 0],
                         [1, 3],
                         [1.5, 1.5],
                         [-2.5, 3],
                         [0, 2],
                         [1, -1],
                         [3, -2]],
                        index=['S', 'A', 'B', 'C', 'D', 'E', 'F'], columns=['x', 'y'])
pcost_f, pcost_g = 2, 1.2  # 单位距离满载, 空载耗能

df, V, F, m_f, G, m_g, L, h_gs, b_vl = data_pre(order, location, pcost_f, pcost_g)
# %% 可视化
print('所有节点V：', V)
print('所有路段L：', L)
# print(L.index('BC'))
# for x in G:
#    print(L.index(x))
print('满载路段F：', F)
print('满载消耗m_f：')
print(m_f)
# print(m_f[2])
print('空载路段G：', G)
# print(G[1])
print('空载消耗m_g：')
print(m_g)
# print('关联矩阵b_vl：\n', b_vl)
# print(b_vl.iloc[1,2],b_vl.iloc[2,1])
print(b_vl.loc['S', 'SA'], b_vl.loc['S', 'BS'])
# print(b_vl.loc['S', 'SA'], b_vl.loc['S', 'BS'])
order_visualise(V, location, F)

# %% 开始求解
# 初始化解

R = pd.DataFrame([[0] * len(L) for _ in range(len(F))], columns=L)
for i in range(len(order)):
    laden_sect: str = F[i]
    idle_sect_0 = 'S' + laden_sect[0]
    idle_sect_1 = laden_sect[1] + 'S'
    R.loc[i, [laden_sect, idle_sect_0, idle_sect_1]] = 1

iter_num = 1
while True:
    quit_loop = False
    # 求解主问题
    mp = MasterProblem(R=R, F=F, G=G, mu=mu, m_f=m_f, m_g=m_g, d_f=df)
    mp.create_model()
    mp.set_objective()
    mp.solve()
    pi = mp.get_dual_vars()
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
        new_route = sub_prob.get_solution()
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
        break

# 最终解可视化
for i in solution_info.keys():
    route_visualise(i, R, location, F, G)
