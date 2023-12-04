import numpy as np
import pandas as pd
import re


# 求两点距离函数
def coordi2distance(a, b):
    return ((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** 0.5


def data_pre(order: pd.DataFrame, location: pd.DataFrame, pcost_f: float, pcost_g: float):
    """
    :param order: 订单集合, index=['AB', 'CD']
    :param location: 运输节点集合
    :param pcost_f: 单位距离满载消耗
    :param pcost_g: 单位距离空载消耗
    :return:
    V:[str] 运输节点集合
    F:[str] 订单集合，满载运输弧集合
    G:[str] 空载运输弧集合
    L:[str] 弧集合，L = F or G
    d_f:Series 订单运输需求, index=F, 例如index=['AB', 'DC'] 
    m_f:Series 满载运输弧的运输电量消耗, index=F 
    m_g:Series 空载运输弧的运输电量消耗, index=G
    b_vl:pd.DataFrame 关联矩阵，index=V, name=L
    # 如果有时间，增加一个路径的可视化
    """
    # df = order['weight']
    s, e = set(order['start']), set(order['end'])
    V = list({'S'} | s | e)  # 所有结点
    # 满载路段
    F, m_f = [], []
    for __, row in order.iterrows():
        F.append(row['start'] + row['end'])
        p1 = location.loc[row['start']]
        p2 = location.loc[row['end']]
        m_f.append(pcost_f * coordi2distance(p1, p2))
    m_f = pd.Series(m_f, index=F)
    # 订单需求
    df = pd.Series(order['weight'], index=F)
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
    m_g = pd.Series(m_g, index=G)
    L = F + G  # 所有路段
    # b_vl 
    b_vl = pd.DataFrame(columns=L, index=V)
    for v in V:
        for x in L:
            m = re.search(v, x)
            if m is None:
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
    pass
