import matplotlib.pyplot as plt
from itertools import combinations

def order_visualise(V,location,F):
'''
output：所有满载路段及空载路段
input：
  V list 所有结点
  location 结点坐标 dataframe
  F list 满载路段
'''
    fig, ax = plt.subplots()
    plt.scatter(location['x'],location['y'],color='k')
    for row in location.iterrows():
        plt.text(row[1].loc['x']+0.1,row[1].loc['y'],row[0],color='k',fontsize='medium',fontweight='bold')
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

if __name__ == "__main__":
    pass
