import pandas as pd
from data_process import data_pre


# 输入数据
u = 3  # 货车最大载货量
m = 100  # 最大电容量
# 订单
order = pd.DataFrame([['A', 'B', 10],
                      ['A', 'D', 8],
                      ['B', 'C', 13],
                      ['D', 'C', 4]], columns=['start', 'end', 'weight'])
# 货运节点
location = pd.DataFrame([[0, 0],
                         [1, 1],
                         [4, 1],
                         [1.5, -1],
                         [5, -2]],
                        index=['S', 'A', 'B', 'C', 'D'], columns=['x', 'y'])
pcost_f, pcost_g = 2, 1.2  # 单位距离满载, 空载耗能

df, V, F, m_f, G, m_g, L, h_gs, b_vl = data_pre(order, location, pcost_f, pcost_g)
