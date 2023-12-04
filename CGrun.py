import pandas as pd

from data_process import data_pre
from Subproblem import SubProblem
from Master_prob2 import MasterProblem
from visualise import order_visualise

# 输入数据
mu = 3  # 货车最大载货量
m = 100  # 最大电容量
# 订单
order = pd.DataFrame([['A', 'B', 20],
                      ['C', 'D', 24]], columns=['start', 'end', 'weight'])
# 货运节点
location = pd.DataFrame([[0, 0],
                         [1, 1],
                         [4, 1],
                         [1.5, -1],
                         [5, -2]],
                        index=['S', 'A', 'B', 'C', 'D'], columns=['x', 'y'])
pcost_f, pcost_g = 2, 1.2  # 单位距离满载, 空载耗能

df, V, F, m_f, G, m_g, L, h_gs, b_vl = data_pre(order, location, pcost_f, pcost_g)
# %% 可视化
print('所有节点V：', V)
print('所有路段L：', L)
# print(L.index('BC'))
# for x in G:
#    print(L.index(x))
print('满载路段F：', F)
print('满载消耗m_f：\n', m_f)
# print(m_f[2])
print('空载路段G：', G)
# print(G[1])
print('空载消耗m_g：\n', m_g)
print('关联矩阵b_vl：\n', b_vl)
# print(b_vl.iloc[1,2],b_vl.iloc[2,1])
print(b_vl.loc['S', 'SA'], b_vl.loc['S', 'BS'])
# print(b_vl.loc['S', 'SA'], b_vl.loc['S', 'BS'])
order_visualise(V, location, F)

# %% 开始求解
# 初始化解
R = pd.DataFrame([[1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]],
                 columns=L)
iter_num = 0
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
    if sub_prob.get_obj*mu <= 1:
        quit_loop = True
    else:
        new_route = sub_prob.get_solution()
        list_R = R.to_numpy().tolist()
        if new_route not in list_R:
            print("Add a new route:")
            print(pd.Series(new_route, index=L))
            R.loc[len(R.index)] = new_route
        else:
            print("Route repeat")
            quit_loop = True
    iter_num += 1
    if quit_loop:
        mp = MasterProblem(R=R, F=F, G=G, mu=mu, m_f=m_f, m_g=m_g, d_f=df)
        mp.create_model(relaxation=False)
        mp.set_objective()
        mp.solve()
        print("SOLVE DONE.")
        print(f"OPT={mp.opt}")
        print(f"SOLUTION={mp.solution}")
        break
