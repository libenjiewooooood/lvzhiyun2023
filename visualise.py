import matplotlib.pyplot as plt
import math
from itertools import combinations
from matplotlib.patches import Arc
from matplotlib.patches import FancyArrow
from data_process import coordi2distance


def visualise_point(location):
    plt.scatter(location['x'], location['y'], color='k')
    for row in location.iterrows():
        plt.text(row[1].loc['x'] + 0.1, row[1].loc['y'], row[0], color='k', fontsize='medium', fontweight='bold')


# define arc class with arrow
class elliptic_arc():
    def __init__(self, p1, p2, arr_len, arr_width, arr_col, h=0.7):
        self.p1 = p1
        self.p2 = p2
        self.center = (p1 + p2) / 2
        self.width = coordi2distance(self.p1, self.p2)
        self.h = h
        self.arr_len = arr_len
        self.arr_width = arr_width
        self.arc_col = arr_col
        self.arr_col = arr_col

    def plot_arc(self, ax):
        rad = self.calculate_rad(self.p1, self.p2)
        angle = 90 - rad * 180 / math.pi
        a = Arc(self.center, self.width, height=self.h, angle=angle, theta2=180, color=self.arc_col, label='fullload')
        ax.add_patch(a)
        return a

    def plot_arrow(self, ax):
        rad = self.calculate_rad(self.p1, self.p2)
        mid = self.get_midpoint
        arr = FancyArrow(mid[0], mid[1], self.arr_len * math.cos(rad - math.pi / 2),
                         -self.arr_len * math.sin(rad - math.pi / 2),
                         width=self.arr_width, length_includes_head=True, head_length=self.arr_len, fc=self.arr_col)
        ax.add_patch(arr)

    @staticmethod
    def coordi2distance(a, b):
        return ((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2) ** 0.5

    @staticmethod
    def calculate_rad(p1, p2):
        l = list(p2 - p1)
        rad = math.atan2(l[0], l[1])
        return rad

    @property
    def get_midpoint(self):
        rad = self.calculate_rad(self.p1, self.p2)
        mid = [self.center[0] + self.h / 2 * math.sin(rad - math.pi / 2),
               self.center[1] + self.h / 2 * math.cos(rad - math.pi / 2)]
        return mid


def order_visualise(V, location, F):
    """
    func：可视化订单
    output：所有满载路段及空载路段
    input：
      V list 所有结点
      location 结点坐标 dataframe
      F list 满载路段
    """
    fig, ax = plt.subplots()
    visualise_point(location)
    for p1, p2 in combinations(V, 2):
        ax1 = ax.plot(location.loc[[p1, p2], 'x'], location.loc[[p1, p2], 'y'], color='b', alpha=0.5, ls='--',
                      label='unload')
    # plt.legend(handles=ax1)
    for f in F:
        r = list(f)
        #    plt.plot(location.loc[[r[0],r[1]],'x'],location.loc[[r[0],r[1]],'y'],color='r')
        line2 = plt.arrow(location.loc[r[0], 'x'], location.loc[r[0], 'y'],
                          location.loc[r[1], 'x'] - location.loc[r[0], 'x'],
                          location.loc[r[1], 'y'] - location.loc[r[0], 'y'], length_includes_head=True, width=0.08,
                          head_width=0.2, label='fullload')
    plt.legend(handles=[ax1[0], line2], loc='best')
    plt.title('test:order', fontsize='x-large', fontweight='bold')
    plt.show()


# 可视化某条路线
def route_visualise(i, R, location, F, G, arr_len=0.1, arr_width=0.05, g_arr_col='b', f_arr_col='r'):
    """
    func:可视化路线
    input：
     i ：路线在R中的序号
     R：路线
     location：结点坐标
     F，G：list 满载/空载路段
    """
    fig, ax = plt.subplots()
    # 绘制结点
    visualise_point(location)
    # 满载
    for f in F:
        if f in R.columns:
            r_f = R.iloc[i, R.columns.get_loc(f)]
            if r_f > 0:
                r = list(f)
                arc1 = elliptic_arc(location.loc[r[0]], location.loc[r[1]], arr_len=arr_len, arr_width=arr_width,
                                    arr_col=f_arr_col)
                farc = arc1.plot_arc(ax)
                arc1.plot_arrow(ax)
                # 标注路段频次
                note_p = arc1.get_midpoint
                plt.text(note_p[0] + 0.1, note_p[1], r_f, color='k', fontsize='medium', fontweight='bold')
        else:
            continue
    # 空载路段
    for g in G:
        if g in R.columns:
            r_g = R.iloc[i, R.columns.get_loc(g)]
            if r_g > 0:
                r = list(g)
                gline = plt.plot(location.loc[[r[0], r[1]], 'x'], location.loc[[r[0], r[1]], 'y'], c='b',
                                 label='unload')
                mid = location.loc[[r[0], r[1]]].sum() / 2
                l = list(location.loc[r[1]] - location.loc[r[0]])
                rad = math.atan2(l[0], l[1])
                arr = FancyArrow(mid[0], mid[1], arr_len * math.cos(rad - math.pi / 2),
                                 -arr_len * math.sin(rad - math.pi / 2),
                                 width=arr_width, length_includes_head=True, head_length=arr_len, fc=g_arr_col)
                ax.add_patch(arr)
                plt.text(mid[0] + 0.1, mid[1], r_g, color='k', fontsize='medium', fontweight='bold')
        else:
            continue
    ax.grid()
    ax.set_xlim(min(location['x']) - 1, max(location['x']) + 1)
    ax.set_ylim(min(location['y']) - 1, max(location['y']) + 1)
    ax.axis('equal')
    fig.suptitle(f'route {i}')
    plt.show()


if __name__ == "__main__":
    # 可视化订单
    # order_visualise(V,location,F)
    # 可视化R中的第1和第3条路线
    # Index_R=[0,2]
    # for i in Index_R:
    #   route_visualise(i,R,location,F,G)
    pass
