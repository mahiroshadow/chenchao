"""
环境的创建
UAV(unmanned aerial vehicles)：无人机
UAVTASKENV：类说明
fog(parameter)：雾设备位置
dist(function)：计算两点之间的欧式距离
connected(function)：计算UE与哪个UAV进行连接
fog device：雾设备，可以直观的理解云的下一层设备
channelGain(function)：信道增益随着物体移动而改变，静止场景下几乎不变，这里只考虑距离带来的影响
SNR(function)：信噪比
reset(function)：初始化无人机设备位置
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class UAVTASKENV():
  def __init__(self, c1, c2):
    self.k = plt.subplots()
    self.fig = self.k[0]
    self.ax = self.k[1]
    # 用户设备聚类1
    self.cluster1 = c1
    # 用户设备聚类2
    self.cluster2 = c2
    # 合并用户聚类
    self.uecord = self.cluster1+self.cluster2


    self.fog = [[450,450,100],[50,50,100]]
    

    self.ue_data = self.UE_data()
    # UE和UAV的连接状况[-1,-1,-1,...-1] 20
    self.connected_array = np.full(20,-1)

    self.ca = np.full(10,0)
    self.ca1 = np.full(10,1)
    self.checkpoint =  np.concatenate((self.ca, self.ca1))

    self.P = 5



  def dist(self, pos1, pos2):
    distance = ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2+(pos1[2]-pos2[2])**2)**0.5
    return distance

  # 若UAE与UE距离小于300，则UE与UAV相连接
  def connected(self, pos_ue, pos_uav, i, j):
    if(self.dist(pos_ue, pos_uav)<=300):
      if(self.connected_array[j]==-1):
        self.connected_array[j] = i
  # 信道增益
  def channelGain(self, d):
    return -50 - 20*(math.log10(d))
  
  # 信噪比
  def SNR(self, gain, Transmitpower):
    P = Transmitpower*10**(gain/10)
    N = 10**(-13)
    return P/N
  
  # 使用香浓定律计算数据传输率（理想情况下，其实算的是最大数据传输率）
  def datarate(self, SNR):
    return math.log2(1+SNR)


  def step(self, actions):
    for i, state in enumerate(self.state):
      action = actions[i]
      # d:前进方向
      d = (action[0])*270
      # theta:前进角度
      theta = (action[1])*180
      dummy = []
      dummy.append(state[0])
      dummy.append(state[1])
      dummy = np.array(dummy, dtype = np.float32)
      # 更新无人机的状态
      state[0] += d*math.cos(math.radians(theta))
      state[1] += d*math.sin(math.radians(theta))
      # 如果UAV飞出了范围({(x,y)|0=<x<=600,0=<y<=600})，重置UAV坐标
      # print(dummy)
      if((state[0]>600 or state[0]<0 )or (state[1]>600 or state[1]<0)):
        state[0] = dummy[0]
        state[1] = dummy[1]
      # 更新UAV与UE的连接状态 
      for j in range(0,len(self.uecord)):
        self.connected(self.uecord[j], state, i, j)

    U = 0
    for i, state in enumerate(self.state):
      action = actions[i]
      X = []
      T = 0
      E = 0
      th = 100000000
      # 取出后面20维向量,像是指代UAV跟UE之间的关系
      for j in range(2,22):
        X.append(action[j]/2 + 0.5)

      for j in range(0,20):
        uav = state
        ue = self.uecord[j]
        dis = self.dist(ue, uav)
        C = self.ue_data[j][0]
        D = self.ue_data[j][1]
        # 计算当前UAV与UE传输速率
        r1 = self.datarate(self.SNR(self.channelGain(dis), 0.1))
        # 计算一个数据传输延迟
        t1 = D/r1
        # 计算每一个任务所需能耗e=p*t(在一定传输时延下所需要消耗的总体能耗)
        e1 = 0.1*t1
        
       
        # 这里的3指代的是UAV分配给UE的计算资源f(大小固定),x指代任务的分配比例,x%任务分配给UAV,(1-x%)分配给fog device
        t2 = (0.001*(x*C*D))/3
        # UAV在处理这些数据D[j]所需要耗费的能量
        e2 = 0.3*t2

        # 下面是计算(1-x%)的任务给fog device所需要的传输时延和计算时延,并计算在该时延下fog device所需产生的能量
        if(x==1):
          th = min(th, r1)
        dis2 = self.dist(uav, self.fog[i])
        r2 = self.datarate(self.SNR(self.channelGain(dis2), 5))
        t3 = (1-x)*D/r2
        e3 = 5*t3
        t4 = 0.0001*((1-x)*C*D)
        # ---------
        if(x!=1):
          th = min(th, min(r1,r2))

        # 这里设置这个factor是因为，如果当前UAV与UE并不相连的话所需要消耗的能量和时延都需要翻倍计算
        factor = 10

        if(self.connected_array[j]==i):
          factor = 1

        t_X = t1+t2+t3+t4
        e_X = e1+e2+e3

        T += factor*t_X
        E += factor*e_X

      # 这边好理解，整体的优化目标为优化UAV和EC所需要消耗的能量和整体的时延，并最大化吞吐量
      U += ((E+T)+(1/th))
    

    # 判断有多少UAV断开与UE的连接
    ct = 0
    for x in self.connected_array:
      if(x==-1):
        ct += 1

    #U = U*ct
    #2 ways to calculate coverage
    #either by multiplying the count of disconnected UAVs in reward itsef, or by giving penalty
    #like
    if(ct>6):
      U += 10*ct
    return self.state, -1*U, False, {}


  # UE_data指的是每一个UE设备上正在运行的服务的大小和所需能耗
  def UE_data(self):
    # 每一个D[j]所需要耗费能量C[j]
    C = []
    # 每一个UE当前任务大小D[j]
    D = [] 
    L = []
    while(len(C)<20):
      C.append(random.randint(100,200))
    while(len(D)<20):
      D.append(random.randint(1,5))
    while(len(L)<20):
      L.append(1)
    k = []
    i = 0
    while(i<20):
      k.append([C[i],D[i],L[i]])
      i += 1
    self.statex = np.array(k, dtype = np.float32)
    return self.statex
  
  # reset设置UAV的初始状态
  def reset(self):
    self.connected_array = np.full(20,-1)
    self.ue_data = self.UE_data()
    X = round(random.uniform(270,300),2)
    Y = round(random.uniform(295,300),2)
    X1 = round(random.uniform(270,300),2)
    Y1 = round(random.uniform(295,300),2)
    if(X1==X):
      #  X1==X and X1-X<=20
      while(X1==X or X1-X<=20):
        X1 = round(random.uniform(300,400),2)

    if(Y1==Y):
      while(Y1==Y):
        Y1 = round(random.uniform(300,400),2)

    u1 = np.array([X,Y,60], dtype = np.float32)
    u2 = np.array([X1,Y1,60], dtype = np.float32)
    # self.state = [u1,u2]
    self.state = [u1]
    return self.state

  def render(self):
    self.ax.cla()
    X = []
    Y = []
    for i in range(0,10):
        X.append(self.cluster1[i][0])
        Y.append(self.cluster1[i][1])

    for i in range(0,10):
        X.append(self.cluster2[i][0])
        Y.append(self.cluster2[i][1])
    self.ax.set_xlim([0,600])
    self.ax.set_ylim([0,600])
    self.ax.set_title('UE-Clusters')
    self.ax.plot(np.array(X),np.array(Y),'ro')
    uavcord1 = (self.state[0][0],self.state[0][1])
    uavcord2 = (self.state[1][0],self.state[1][1])
    self.ax.scatter(uavcord1[0], uavcord1[1], marker = "X", s= 100)
    self.ax.scatter(uavcord2[0], uavcord2[1], marker = "X", s= 100)
    self.ax.scatter(self.fog[0][0], self.fog[0][1], marker = "^", s= 100)
    self.ax.scatter(self.fog[1][0], self.fog[1][1], marker = "^", s= 100)
    self.ax.grid(True)
    plt.show()
    plt.pause(0.8)


"""
UE(User Equipment)： 用户设备
每个用户设备含有一个坐标：{(x,y,z)∈R^3|x∈(x1,x2), y∈(y1,y2), z=0} 
"""
def create_UE_cluster(x1, y1, x2, y2):
  X = []
  Y = []
  Z = []
  while(len(X)<10):
    cord_x = round(random.uniform(x1,x2),2)
    if(cord_x not in X):
      X.append(cord_x)
  while(len(Y)<10):
    cord_y = round(random.uniform(y1,y2),2)
    if(cord_y not in Y):
      Y.append(cord_y)
  while(len(Z)<10):
      Z.append(0)
  k = []
  i = 0
  while(i<10):
      k.append([X[i],Y[i],Z[i]])
      i += 1
  return k


