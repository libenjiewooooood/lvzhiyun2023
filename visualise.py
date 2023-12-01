import matplotlib.pyplot as plt
import math
from matplotlib import patches
from matplotlib import path 
from matplotlib.path import Path
from itertools import combinations
from data_process import coordi2distance

def visualise_point(location):
    plt.scatter(location['x'],location['y'],color='k')
    for row in location.iterrows():
        plt.text(row[1].loc['x']+0.1,row[1].loc['y'],row[0],color='k',fontsize='medium',fontweight='bold')
def get_curve(location,p1,p2,f):
    '''
    p1,p2='A','B'
    location dataframe 
    '''
    code=([Path.MOVETO,Path.CURVE3,Path.CURVE3])
    width=coordi2distance(location.loc[p1],location.loc[p2])
    control_point=location.loc[[p1,p2]].sum()/2+width*0.2
    vert=tuple(location.loc[p1]),tuple(control_point),tuple(location.loc[p2])
    plt.text(control_point['x']-width*0.1,control_point['y']-width*0.1,f,color='k',fontsize='medium',fontweight='bold')
    return code,vert
def note_freq(location,p1,p2,f):
    '''
    func:标注路段执行次数
    
    '''
    control_point=location.loc[[p1,p2]].sum()/2
    width=coordi2distance(location.loc[p1],location.loc[p2])
    plt.text(control_point['x'],control_point['y'],f,color='k',fontsize='medium',fontweight='bold')

def order_visualise(V,location,F):
'''
func：可视化订单
output：所有满载路段及空载路段
input：
  V list 所有结点
  location 结点坐标 dataframe
  F list 满载路段
'''
    fig, ax = plt.subplots()
    visualise_point(location)
    for p1,p2 in combinations(V,2):
        ax1=ax.plot(location.loc[[p1,p2],'x'],location.loc[[p1,p2],'y'],color='b',alpha=0.5,ls='--',label='unload')
    # plt.legend(handles=ax1)
    for f in F:
        r=list(f)
    #    plt.plot(location.loc[[r[0],r[1]],'x'],location.loc[[r[0],r[1]],'y'],color='r')
        line2=plt.arrow(location.loc[r[0],'x'],location.loc[r[0],'y'],location.loc[r[1],'x']-location.loc[r[0],'x'],
                  location.loc[r[1],'y']-location.loc[r[0],'y'],length_includes_head=True,width=0.08,head_width=0.2,label='fullload')
    plt.legend(handles=[ax1[0],line2],loc='best')
    plt.title('test:order',fontsize='x-large',fontweight='bold')
    plt.show()

# 可视化某条路线
def route_visualise(i,R,location,F,G,):
  '''
  func:可视化路线
  input：
   i ：路线在R中的序号
   R：路线
   location：结点坐标
   F，G：list 满载/空载路段
  '''
    fig, ax = plt.subplots()
    # 绘制结点
    visualise_point(location)
    codes,verts=[],[]
    # 满载
    for f in F: 
        if f in R.columns:
            r_f=R.iloc[i,R.columns.get_loc(f)]
            if r_f>0:
                r=list(f)
                code,vert=get_curve(location,r[0],r[1],r_f)            
                codes.extend(code)
                verts.extend(vert)
        else:
            continue
    path = Path(verts, codes)
    patch = patches.PathPatch(path, ec='r',facecolor='none',linewidth=2)
    ax.add_patch(patch)
    # 空载路段
    for g in G:
        if g in R.columns:
            r_g=R.iloc[i,R.columns.get_loc(g)]
            if r_g>0:
                r=list(g)
                plt.arrow(location.loc[r[0],'x'],location.loc[r[0],'y'],location.loc[r[1],'x']-location.loc[r[0],'x'],
                  location.loc[r[1],'y']-location.loc[r[0],'y'],length_includes_head=True,width=0.01,head_width=0.1)
                note_freq(location,r[0],r[1],r_g)
        else:
            continue    
    ax.grid()
    ax.set_xlim(min(location['x'])-1, max(location['x'])+1)
    ax.set_ylim(min(location['y'])-1, max(location['y'])+1)
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
