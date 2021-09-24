import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

'''
SVM
    优化目标：分类正确的前提下，最大化支持向量到超平面的间隔，可转换为 min(w,b) 1/2(||w||^2) s.t. y(w*x+b)>=1
'''

class SVM(object):
    def __init__(self,max_it=10,C=0.5):
        '''
        it : 最大无效迭代次数，默认10，即连续10次无效更新则停止优化过程
        C : 松弛因子，默认为0.5
        '''
        self.max_it = max_it
        self.C = C

    def find_ij(self,size):
        '''
        根据是否满足KKT条件以及误差最大化原则进行启发式的i和j查找
        '''
        return random.sample(range(size),2)

    def f(self,x):
        '''
        f(x) = sum(λi*yi*K(xi,x))+b
        '''
        return sum([li*yi*np.dot(xi,x) for li,yi,xi in zip(self.lambdas,self.y,self.X)])+self.b

    def predict(self,x):
        '''
        预测函数，f(x) = sum(λi*yi*K(xi,x))+b
        '''
        return int(np.sign(self.f(x)))

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
        it,count = 0,0
        while it < self.max_it:
            ## 确定i，j
            i,j = self.find_ij(self.size)
            ## 更新λi和λj
            xi,yi,li_old,xj,yj,lj_old = X[i],y[i],self.lambdas[i],X[j],y[j],self.lambdas[j]
            Ei,Ej = self.f(xi)-yi,self.f(xj)-yj
            eta = np.dot(xi,xi)+np.dot(xj,xj)-2*np.dot(xi,xj)
            if eta <= 0:
                print("warning eta=0.0")
                continue
            L = max(0,lj_old-li_old) if yi!=yj else max(0,li_old+lj_old-self.C)
            H = min(self.C,self.C+lj_old-li_old) if yi!=yj else min(self.C,li_old+lj_old)
            lj_new = np.clip(lj_old-(yj*(Ej-Ei))/(eta),L,H)
            li_new = li_old+(lj_old*yj/yi)-(lj_new*yj/yi)
            if abs(li_new - li_old)<0.000001 or abs(lj_new - lj_old)<0.000001:
                it += 1
                continue
            it = 0
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
            count += 1

        ## 计算w
        self.w = sum([li*yi*xi for li,yi,xi in zip(self.lambdas,self.y,self.X)])

        return count,self.lambdas,self.w,self.b


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data[iris.target<2,:2]
    y = iris.target[iris.target<2]
    y[y==0] = -1

    for pos,max_it,C in [[331,100,.5],[332,1000,.5],[333,1000,1],[334,1000,10],[335,1000,100],[336,1000,1000],[337,1000,10],[338,1000,100],[339,1000,1000]]:
        print(pos,max_it,C)
        plt.subplot(pos)
        # 指定坐标轴范围
        xx = [x for x in [min([z[0] for z in X])-1,max([z[0] for z in X])+1]]
        yy = [x for x in [min([z[1] for z in X])-1,max([z[1] for z in X])+1]]
        plt.xlim(xx)
        plt.ylim(yy)

        # 绘制数据点
        plt.scatter([x[0] for x in X[y==1]],[x[1] for x in X[y==1]],marker="o")
        plt.scatter([x[0] for x in X[y==-1]],[x[1] for x in X[y==-1]],marker="x")

        model = SVM(max_it=max_it,C=C)
        count,l,w,b = model.train(X,y)

        # 绘制分割平面
        xx = [x for x in [min([z[0] for z in X]),max([z[0] for z in X])]]
        yy = [(-b-w[0]*x)/w[1] for x in xx]
        plt.plot(xx,yy)

        # 绘制支持向量点
        xx = [xi[0] for xi,li in zip(X,l) if li>0.1]
        yy = [xi[1] for xi,li in zip(X,l) if li>0.1]
        plt.scatter(xx,yy,marker="o",color="",edgecolor="black",s=100)
        plt.title(f"max_it={max_it} C={C} count={count}")

    plt.show()
