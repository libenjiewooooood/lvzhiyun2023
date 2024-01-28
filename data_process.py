import numpy as np
import pandas as pd
import re


# 求两点距离函数
def coordi2distance(a, b):
    return ((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** 0.5


def data_pre(order: pd.DataFrame, location: pd.DataFrame, pcost_f: float, pcost_g: float, S: set, Se: set):
    """
    :param order: 订单集合, index=['AB', 'CD']
    :param location: 运输节点集合
    :param pcost_f: 单位距离满载消耗
    :param pcost_g: 单位距离空载消耗
    :param S 换电站集合
    :param Se 车库集合
    :return:
        V:[str] 所有节点集合
        Vs:[str] 订单起点集合
        F:[str] 订单集合，满载运输弧集合
        G:[str] 空载运输弧集合
        L:[str] 弧集合，L = F or G
        df:Series 订单运输需求, index=F, 例如index=['AB', 'DC']
        m_f:Series 满载运输弧的运输电量消耗, index=F
        m_g:Series 空载运输弧的运输电量消耗, index=G


    """
    # df = order['weight']
    s, e = set(order['start']), set(order['end'])
    V = list(S | s | e | Se)
    Vs = set(order['start'])  # 订单起点集合
    # 满载路段
    F, m_f = [], []
    for __, row in order.iterrows():
        F.append(row['start'] + row['end'])
        p1 = location.loc[row['start']]
        p2 = location.loc[row['end']]
        m_f.append(pcost_f * coordi2distance(p1, p2))
    m_f = pd.Series(m_f, index=F)
    # 订单需求
    df = pd.Series(list(order['weight']), index=F)
    # 空载路段
    G, m_g = [], []

    for y in S:
        p1 = location.loc[y]
        for x in s:
            p2 = location.loc[x]
            G.append(y + x)
            m_g.append(pcost_g * coordi2distance(p1, p2))
        for x in e:
            p2 = location.loc[x]
            G.append(x + y)
            m_g.append(pcost_g * coordi2distance(p1, p2))
        for x in Se:
            p2 = location.loc[x]
            G.append(x + y)
            m_g.append(pcost_g * coordi2distance(p1, p2))
    for x in e:
        for y in s:
            if x != y:
                G.append(x + y)
                p1 = location.loc[y]
                p2 = location.loc[x]
                m_g.append(pcost_g * coordi2distance(p1, p2))
    for x in Se:
        for y in s:
            G.append(x + y)
            p1 = location.loc[y]
            p2 = location.loc[x]
            m_g.append(pcost_g * coordi2distance(p1, p2))
        for y in e:
            G.append(y + x)
            p1 = location.loc[y]
            p2 = location.loc[x]
            m_g.append(pcost_g * coordi2distance(p1, p2))

    m_g = pd.Series(m_g, index=G)
    L = F + G  # 所有路段
    return df, V, Vs, F, m_f, G, m_g, L


if __name__ == "__main__":
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
