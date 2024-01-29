import numpy as np
import pandas as pd
import re


# 求两点距离函数
def coordi2distance(a, b):
    return ((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** 0.5


def data_pre(order: pd.DataFrame, location: pd.DataFrame, pcost_f: float, pcost_g: float, S:set, Se:set):
    """
    :param order: 订单集合, index=['AB', 'CD']
    :param location: 运输节点集合
    :param pcost_f: 单位距离满载消耗
    :param pcost_g: 单位距离空载消耗
    :param is_e 说明是否为可换点情景
    :Se 换电站集合
    :S 车库集合
    :return:
    V:[str] 所有节点集合
    F:[str] 订单集合，满载运输弧集合
    G:[str] 空载运输弧集合
    L:[str] 弧集合，L = F or G
    d_f:Series 订单运输需求, index=F, 例如index=['AB', 'DC'] 
    m_f:Series 满载运输弧的运输电量消耗, index=F 
    m_g:Series 空载运输弧的运输电量消耗, index=G
    b_vl:pd.DataFrame 关联矩阵，index=V, name=L

    """
    # df = order['weight']
    s, e = set(order['start']), set(order['end'])
    V = list(S | s | e | Se)
    Vs = list(order['start'])
    Vs_all = [f"{node}{i}" for node in Vs for i in range(1, 4)]
    V_all = list(set(Vs_all)|S|Se)
    #print("Vs_all")
    #print(Vs_all)
    #print("V_all")
    #print(V_all)
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
    return df, V, Vs, F, m_f, G, m_g, L, V_all


if __name__ == "__main__":
    u = 3  # 货车最大载货量
    order = pd.DataFrame([['A', 'B', 10], ['C', 'D', 13]],
                         columns=['start', 'end', 'weight'])
    location = pd.DataFrame([[-2, 0], [2,1],[1, 1], [4, 1], [1.5, -1], [5, -2], [0, 1], [2, 0]],
                            index=['S','s', 'A', 'B', 'C', 'D', 'E', 'F'], columns=['x', 'y'])
    pcost_f, pcost_g = 1, 0.8  # 单位距离满载/空载耗能
    # 换电站集合
    Se = {'E', 'F'}
    S = {'S','s'}
    #S = {'s'}
    df,V,Vs,F,m_f,G,m_g,L,V_all=data_pre(order,location,pcost_f,pcost_g,S, Se)

    print("订单需求df")
    print(df)
    print("所有节点V")
    print(V)
    print("所有起点节点Vs")
    print(Vs)
    print("订单集合F")
    print(F)
    print("满载路段电量消耗m_f")
    print(m_f)
    print("空载路段集合G")
    print(G)
    print("空载路段电量消耗m_g")
    print(m_g)
    print("弧集合L")
    print(L)
    print("V_all")
    print(V_all)
    q = pd.DataFrame(0.0, index=V_all, columns=V_all)
    #print(q)
    # 为q赋值
    for i in V_all:
        for j in V_all:
            if i != j:
                if i[0] in Vs:
                    # 为 m_f 找到以 i[0] 开头的键
                    key_f = next((k for k in m_f.keys() if k.startswith(i[0])), None)

                    # 为 m_g 找到以 j 结尾的键
                    key_g = next((k for k in m_g.keys() if k.startswith(key_f[1]) and k.endswith(j[0])), None)

                    # 从 m_f 和 m_g 中获取对应的电量消耗值
                    consumption_f = m_f.get(key_f, 0)  # 如果键不存在，返回0
                    consumption_g = m_g.get(key_g, 0)  # 如果键不存在，返回0
                else:
                    key_f = 0
                    key_g = next((k for k in m_g.keys() if k.startswith(i[0]) and k.endswith(j[0])), None)
                    consumption_f = key_f  # 如果键不存在，返回0
                    consumption_g = m_g.get(key_g, 0)  # 如果键不存在，返回0
                # 为 q 的对应位置赋值
                q.at[i,j] = consumption_f + consumption_g
                # if i == 'F' and j == 'A1':
                #     print('q(F,A1)')
                #     print(i[0],j[0])
                #     print(q[i][j])
    for i in V_all:
        for j in V_all:
            if q.at[i,j] == 0:
                q.at[i,j] = 10
    print(q)
    print(q['A1']['A2'])
    print('q[A1][C1]')
    print(q['A1']['C1'])
    print('q.at[A1, C1]')
    print(q.at['A1', 'C1'])
