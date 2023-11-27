import numpy as np
import pandas as pd
import re

# 输入数据
u = 3  # 货车最大载货量
order = pd.DataFrame([['A', 'B', 10], ['A', 'D', 8], ['B', 'C', 13], ['D', 'C', 4]], columns=['start', 'end', 'weight'])
location = pd.DataFrame([[0, 0], [1, 1], [4, 1], [1.5, -1], [5, -2]], index=['S', 'A', 'B', 'C', 'D'],
                        columns=['x', 'y'])
pcost_f, pcost_g = 1, 0.8  # 单位距离满载/空载耗能


# 求两点距离函数
def coordi2distance(a, b):
    return ((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** 0.5


def data_pre(u, order, location, pcost_f, pcost_g):
    df = order['weight']
    s, e = set(order['start']), set(order['end'])
    V = {'S'} | s | e  # 所有结点
    # 满载路段
    F, m_f = [], []
    for __, row in order.iterrows():
        F.append(row['start'] + row['end'])
        p1 = location.loc[row['start']]
        p2 = location.loc[row['end']]
        m_f.append(pcost_f * coordi2distance(p1, p2))
    # 空载路段
    G, m_g = [], []
    for x in s:
        p1 = location.loc['S']
        p2 = location.loc[x]
        G.append('S' + x)
        m_g.append(pcost_g * coordi2distance(p1, p2))
    for x in e:
        p1 = location.loc['S']
        p2 = location.loc[x]
        G.append(x + 'S')
        m_g.append(pcost_g * coordi2distance(p1, p2))
    for x in e:
        for y in s:
            if x != y:
                G.append(x + y)
                p1 = location.loc[y]
                p2 = location.loc[x]
                m_g.append(pcost_g * coordi2distance(p1, p2))
    L = F + G  # 所有路段
    # b_vl 
    b_vl = pd.DataFrame(columns=L, index=V)
    for v in V:
        for x in L:
            m = re.search(v, x)
            if m == None:
                b_vl.loc[v, x] = 0
            elif m.start() == 0:
                b_vl.loc[v, x] = -1
            elif m.start() == 1:
                b_vl.loc[v, x] = 1
                # h_gs s的流入运输路径关联矩阵
    h_gs = b_vl.loc['S'].copy()
    h_gs[h_gs == -1] = 0
    return df, V, F, m_f, G, m_g, L, h_gs, b_vl

if __name__ == "__main__":
    df, V, F, m_f, G, m_g, L, h_gs, b_vl = data_pre(u, order, location, pcost_f, pcost_g)
