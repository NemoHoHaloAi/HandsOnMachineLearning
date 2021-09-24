import numpy as np
import random,time
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from 支持向量机 import SVM

'''
Platt SVM

Platt迭代过程：
    1. 第一个λ选择基于KKT条件，在全量数据与非边界数据集中交替选择迭代；
    2. 第二个λ选择基于其E与上一个λ的E的差值的绝对值大小，越大越好；
'''

class PlattSVM(SVM):
    def __init__(self,max_it=10,C=0.5,tolerance=0.001):
        '''
        max_it : 最大交替迭代次数，类似epochs，一次迭代会遍历全部数据或者全部非边界数据
        C : 松弛因子，默认为0.5
        tolerance : 容忍度，kkt条件是很严谨苛刻的，因此若Ei*yi绝对值小于tolerance则不需要优化
        '''
        self.tolerance = tolerance
        super(PlattSVM,self).__init__(max_it,C)

    def rand_j(self,size,i):
        '''
        固定i之后随机选择一个不等于i的作为j
        '''
        return random.choice(list(range(0,i))+list(range(i+1,size)))

    def find_j(self,size,i):
        '''
        固定i之后，在满足0<λ<C的样本中寻找一个使得ABS(Ei-Ej)最大的j
        '''
        valid_l_list = np.array([k for k,l in enumerate(self.lambdas) if 0 < l < self.C])
        if len(valid_l_list)>1:
            max_e,max_j = 0,-1
            Ei = self.f(self.X[i])-self.y[i]
            for k in valid_l_list:
                if k != i:
                    Ej = self.f(self.X[k])-self.y[k]
                    e = abs(Ej-Ei)
                    if e > max_e:
                        max_e,max_j = e,k
            return max_j
        else:
            return self.rand_j(size,i) 

    def find_ij(self,size,i):
        '''
        根据是否满足KKT条件以及误差最大化原则进行启发式的i和j查找
        '''
        vec = np.array([(self.predict(xi)-1)**2 for xi in self.X])
        vec1,vec2,vec3 = vec.copy(),vec.copy(),vec.copy()
        vec1[(self.lambdas==0) & ([(self.predict(xi)*yi)>=1 for xi,yi in zip(self.X,self.y)])] = 0
        vec2[(self.lambdas<self.C) & (self.lambdas>0) & ([(self.predict(xi)*yi)==1 for xi,yi in zip(self.X,self.y)])] = 0
        vec3[(self.lambdas==self.C) & ([(self.predict(xi)*yi)<=1 for xi,yi in zip(self.X,self.y)])] = 0
        vec4 = vec1 + vec2 + vec3
        i = random.choice([idx for idx,v in enumerate(vec4) if v==max(vec4)])
        j = self.find_j(size,i)
        return i,j

    def do(self,i,j):
        '''
        已知i、j，进行迭代优化
        '''
        ## 更新λi和λj
        xi,yi,li_old,xj,yj,lj_old = X[i],y[i],self.lambdas[i],X[j],y[j],self.lambdas[j]
        Ei,Ej = self.f(xi)-yi,self.f(xj)-yj
        eta = np.dot(xi,xi)+np.dot(xj,xj)-2*np.dot(xi,xj)
        if eta <= 0:
            print("warning eta=0.0")
            return 0
        L = max(0,lj_old-li_old) if yi!=yj else max(0,li_old+lj_old-self.C)
        H = min(self.C,self.C+lj_old-li_old) if yi!=yj else min(self.C,li_old+lj_old)
        lj_new = np.clip(lj_old-(yj*(Ej-Ei))/(eta),L,H)
        li_new = li_old+(lj_old*yj/yi)-(lj_new*yj/yi)
        self.lambdas[i],self.lambdas[j] = li_new,lj_new
        ## 更新b
        bi_new = -Ei+(li_old-li_new)*yi*np.dot(xi,xi)+(lj_old-lj_new)*yj*np.dot(xj,xi)+self.b
        bj_new = -Ej+(li_old-li_new)*yi*np.dot(xi,xj)+(lj_old-lj_new)*yj*np.dot(xj,xj)+self.b
        if 0 < bi_new < self.C:
            self.b = bi_new
        elif 0 < bj_new < self.C:
            self.b = bj_new
        else:
            self.b = (bi_new+bj_new)/2.
        return 1

    def meet_kkt(self,l,x):
        if l == 0:
            return self.f(x) >= 1
        elif l == self.C:
            return self.f(x) <= 1
        else:
            return self.f(x) == 1

    def step(self,i):
        '''
        一个迭代步骤
        '''
        xi,yi,li_old = X[i],y[i],self.lambdas[i]
        Ei = self.f(xi)-yi
        r = Ei*yi
        # 判断是否满足容忍度以及参数λ在可优化范围内
        if ((r < -self.tolerance and li_old < self.C) or (r > self.tolerance and li_old > 0)):# and not self.meet_kkt(li_old,xi):
            j = self.find_j(self.size,i)
            return self.do(i,j)
        return 0

    def train(self,X,y):
        '''
        SMO求解对偶问题
        '''
        # 初始化变量
        self.X,self.y = X,y
        self.size = X.shape[0]
        self.lambdas = np.zeros(self.size)
        self.w = np.zeros(self.size)
        self.b = 0

        # 迭代优化过程
        it,count,entire = 0,0,True
        while it < self.max_it:
            ## 确定i，j
            if entire:
                for i in range(self.size):
                    count += self.step(i)
            else:
                for i in [idx for idx,l in enumerate(self.lambdas) if 0 < l < self.C]:
                    count += self.step(i)
            entire = not entire
            it += 1

        ## 计算w
        self.w = sum([li*yi*xi for li,yi,xi in zip(self.lambdas,self.y,self.X)])

        return count,self.lambdas,self.w,self.b


if __name__ == '__main__':
    iris = load_iris()
    X1 = iris.data[iris.target<2,:2]
    y1 = iris.target[iris.target<2]
    y1[y1==0] = -1

    X2 = np.array([[1,1],[2,2],[3,3],[3,1],[2,3],[1,3],[2.5,2.6]])
    y2 = np.array([1,1,1,1,-1,-1,-1])

    for X,y,pos,M,max_it,C in [[X1,y1,[2,6,1],SVM,1000,.5],[X1,y1,[2,6,2],SVM,1000,10],[X1,y1,[2,6,3],SVM,1000,1000],[X1,y1,[2,6,4],PlattSVM,50,.5],[X1,y1,[2,6,5],PlattSVM,50,10],[X1,y1,[2,6,6],PlattSVM,50,1000],[X2,y2,[2,6,7],SVM,1000,.5],[X2,y2,[2,6,8],SVM,1000,10],[X2,y2,[2,6,9],SVM,1000,1000],[X2,y2,[2,6,10],PlattSVM,50,.5],[X2,y2,[2,6,11],PlattSVM,50,10],[X2,y2,[2,6,12],PlattSVM,50,1000]]:
        name = 'SVM' if M==SVM else 'PlattSVM'
        print(pos,name,max_it,C)
        row,col,idx = pos
        plt.subplot(row,col,idx)
        # 指定坐标轴范围
        xx = [x for x in [min([z[0] for z in X])-1,max([z[0] for z in X])+1]]
        yy = [x for x in [min([z[1] for z in X])-1,max([z[1] for z in X])+1]]
        plt.xlim(xx)
        plt.ylim(yy)

        # 绘制数据点
        plt.scatter([x[0] for x in X[y==1]],[x[1] for x in X[y==1]],marker="o")
        plt.scatter([x[0] for x in X[y==-1]],[x[1] for x in X[y==-1]],marker="x")

        t1 = time.time()
        model = M(max_it=max_it,C=C)
        count,l,w,b = model.train(X,y)
        cost = time.time()-t1

        # 绘制分割平面
        xx = [x for x in [min([z[0] for z in X]),max([z[0] for z in X])]]
        yy = [(-b-w[0]*x)/w[1] for x in xx]
        plt.plot(xx,yy)

        # 绘制支持向量点
        xx = [xi[0] for xi,li in zip(X,l) if li>0.001]
        yy = [xi[1] for xi,li in zip(X,l) if li>0.001]
        plt.scatter(xx,yy,marker="o",color="",edgecolor="black",s=100)
        plt.title(f"{name}:C={C} => cost={cost:>.2f}s")

    plt.show()
