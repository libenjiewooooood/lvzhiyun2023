import pandas as pd
from data_process import data_pre
from Subproblem import SubProblem
from Master_problem import MasterProblem

# 输入数据
u = 3  # 货车最大载货量
m = 100  # 最大电容量
# 订单
order = pd.DataFrame([['A', 'B', 10],
                      ['C', 'D', 14]], columns=['start', 'end', 'weight'])
# 货运节点
location = pd.DataFrame([[0, 0],
                         [1, 1],
                         [4, 1],
                         [1.5, -1],
                         [5, -2]],
                        index=['S', 'A', 'B', 'C', 'D'], columns=['x', 'y'])
pcost_f, pcost_g = 2, 1.2  # 单位距离满载, 空载耗能

df, V, F, m_f, G, m_g, L, h_gs, b_vl = data_pre(order, location, pcost_f, pcost_g)


print('所有节点V：',V)
print('所有路段L：',L)
#print(L.index('BC'))
#for x in G:
#    print(L.index(x))
print('满载路段F：',F)
print('满载消耗m_f：',m_f)
#print(m_f[2])
print('空载路段G：',G)
#print(G[1])
print('空载消耗m_f：',m_g)
print('流矩阵b_vl',b_vl)
#print(b_vl.iloc[1,2],b_vl.iloc[2,1])
print(b_vl.loc['S','SA'],b_vl.loc['S','BS'])
print(b_vl.loc['S','SA'],b_vl.loc['S','BS'])