import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

'''
对于数值型字段，可以假设其分布属于高斯分布，即正态分布，此时只需要计算高斯分布的参数μ和σ^2即可计算各个取值对应的概率；

贝叶斯公式：P(Y=+1|X1=1,X2=1) = ( P(Y=+1) * P(X1=1,X2=1|Y=+1) ) / (P(X1=1,X2=1))
'''

class GaussianNaiveBayes(object):
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def train(self):
        # 先验概率
        self.p_0 = 1.*len(self.y[self.y==0])/len(self.y)
        self.p_1 = 1 - self.p_0

        # 高斯分布参数
        self.means = np.array([np.mean(self.X[:,i]) for i in range(self.X.shape[1])])
        self.means_0 = np.array([np.mean(self.X[self.y==0][:,i]) for i in range(self.X.shape[1])])
        self.means_1 = np.array([np.mean(self.X[self.y==1][:,i]) for i in range(self.X.shape[1])])
        self.stds = np.array([np.std(self.X[:,i]) for i in range(self.X.shape[1])])
        self.stds_0 = np.array([np.std(self.X[self.y==0][:,i]) for i in range(self.X.shape[1])])
        self.stds_1 = np.array([np.std(self.X[self.y==1][:,i]) for i in range(self.X.shape[1])])

    def gaussian(self,mean,std,value):
        '''
        高斯PDF
        '''
        return 1./(std*np.sqrt(2.*np.pi))*np.exp(-((value-mean)**2)/(2*(std**2)))

    def predict(self,x):
        pre_probability = self.p_0
        likelihood = self.gaussian(self.means[0],self.stds[0],x[0]) * self.gaussian(self.means[1],self.stds[1],x[1])
        p = self.gaussian(self.means_0[0],self.stds_0[0],x[0]) * self.gaussian(self.means_0[1],self.stds_0[1],x[1])
        p0 = pre_probability * p / likelihood
        print(pre_probability,p,likelihood,p0)

        pre_probability = self.p_1
        p = self.gaussian(self.means_1[0],self.stds_1[0],x[0]) * self.gaussian(self.means_1[1],self.stds_1[1],x[1])
        p1 = pre_probability * p / likelihood
        print(pre_probability,p,likelihood,p1)

        return p0,p1,0 if p0>p1 else 1

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data[iris.target!=0,2:4]
    y = iris.target[iris.target!=0]
    y[y==1] = 0
    y[y==2] = 1

    model = GaussianNaiveBayes(X,y)
    model.train()
    for i in range(10):
        idx = np.random.randint(X.shape[0])
        print(X[idx],y[idx],model.predict(X[idx]))

    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
    ax = plt.subplot(grid[0, 0])
    sns.histplot(data=X[y==0][:,1],kde=True,color='r',ax=ax)
    sns.histplot(data=X[y==1][:,1],kde=True,color='g',ax=ax)

    ax = plt.subplot(grid[0, 1:])
    resolution = 0.05
    x_min,y_min = min([x[0] for x in X]),min([x[1] for x in X])
    x_max,y_max = max([x[0] for x in X]),max([x[1] for x in X])
    xx1, xx2 = np.mgrid[x_min-resolution:x_max+resolution:resolution, y_min-resolution:y_max+resolution:resolution]
    grid_data = np.c_[xx1.ravel(), xx2.ravel()]
    y_pred = np.array([model.predict(np.array(x))[1]/(model.predict(np.array(x))[0]+model.predict(np.array(x))[1]) for x in grid_data]).reshape(xx1.shape)
    plt.contourf(xx1, xx2, y_pred, 25, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.8)
    plt.scatter([x[0] for x,yi in zip(X,y) if yi==0],[x[1] for x,yi in zip(X,y) if yi==0],marker='o',s=30,color='r',linewidths=1,edgecolors='white',label='flower a')
    plt.scatter([x[0] for x,yi in zip(X,y) if yi==1],[x[1] for x,yi in zip(X,y) if yi==1],marker='*',s=30,color='g',linewidths=1,edgecolors='white',label='flower b')
    plt.legend()

    ax = plt.subplot(grid[1, 1:])
    sns.histplot(data=X[y==0][:,0],kde=True,color='r',ax=ax)
    sns.histplot(data=X[y==1][:,0],kde=True,color='g',ax=ax)

    plt.show()
