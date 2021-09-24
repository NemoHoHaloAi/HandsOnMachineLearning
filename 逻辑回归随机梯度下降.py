import time
import numpy as np
from 线性回归最小二乘法矩阵实现 import LinearRegression as LR
from sklearn.datasets import load_iris

'''
决策边界：
    感知机：决策边界就是类别的分界线，处于错误方向的点即为分类错误点；
    逻辑回归：决策边界表示分为正类和负类均为50%，各点被分为正类的概率直观上由其到决策边界的距离决定；
    以上，对于数据中的噪声，假设噪声点实际为负类，但是被分到正类一侧，如果是感知机，则无法判断，而逻辑回归以概率为基础，如果该噪声点实际被分为正类的概率仅为52%，那么实际上它属于负类的可能性也很大，即逻辑回归认为数据的产生是有一定随机性的，相比于简单的0/1，概率值更能表现其实际情况；


输出函数：
    感知机使用sign作为输出函数：sign(wx+b)
    逻辑回归使用sigmoid作为概率输出函数：sigmoid(wx+b)，sigmoid=1/(1+e^-z)，这里z=wx+b，可以看到当z=0时，也就是处于决策边界时，此时sigmoid=0.5，也就是50%，除此之外，z越大，sigmoid输出越大，可以认为越有可能是正类，反之即为负类，且可以通过极限推导sigmoid区间为(0,1)；


从概率角度看，sigmoid给出了条件x下y=1的概率：
    正类的分布 P(Y=1|X)=S(wx+b)=1/(1+e^-(wx+b))，其中x~P(X)，y~P(Y)
    负类的分布 P(Y=-1|X)=1-S(wx+b)=1-1/(1+e^-(wx+b))
    以上，有 P(Y|X)={
        S(wx+b) , y=+1
        1-S(wx+b) , y=-1
    }
    则 y|x 服从条件分布 P(Y|X)


为什么使用sigmoid：
    sigmoid自身性质：
        1. 输入是整个实数域，输出是(0,1)，输出不包含0和1，使得对于机器学习仅使用整体的一部分作为样本进行训练的场景，很适合处理未出现在样本集中的类别；
        2. 图像曲线对于两侧极值不敏感，二分类的输出符合伯努利分布，输入一般认为符合正态分布，图像曲线也符合这一点；
        3. 易于求导，sigmoid函数的导数为S(wx+b)*(1-S(wx+b))，参数优化过程基本就是求导过程，因此易于求导很重要；
    贝叶斯概率推导：
        sigmoid函数可以由伯努利、正态分布+贝叶斯全概率公式推导得到；


假设空间：H = {h(x) = P(Y=1|X) = Sigmoid(wx+b)}


误差函数：
    感知机：yi*sign(w*xi+b)，yi∈{-1,+1}，模型分类正确返回值为+1，错误返回值为-1，对所有样本进行求和即可得到score值
    逻辑回归：ln(1+e^-(yi*wxi))，yi∈{-1,+1}，模型分类正确返回值>=0，错误返回值<0，且错误越严重（正类：1%，负类：99%），负数越大，对所有样本计算该误差加起来求平均即为逻辑回归的误差函数；
'''

class LogisticRegression(LR):
    def __init__(self,X,y,epochs=5000,eta=0.1,epsilon=0.001,sgd=False):
        super(LogisticRegression,self).__init__(X,y)
        self.epochs = epochs
        self.eta = eta
        self.epsilon = epsilon
        self.sgd = sgd
        self.wk = np.array([0 for i in range(self.X.shape[1])])

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def h(self,x):
        '''
        假设函数
        '''
        return self.sigmoid(x@self.wk.T)

    def drhd(self,X,y,w):
        '''
        梯度下降函数：经验误差函数的梯度
        '''
        ew = []
        for i in range(X.shape[1]):
            ewi = np.mean(-y*X[:, i]*np.exp(-y*(X@w))/(1+np.exp(-y*(X@w))))
            ew.append(ewi)
        return np.array(ew)

    def sdrhd(self,X,y,w):
        '''
        随机梯度下降函数

        随机梯度下降中不再需要设置epsilon，或者说不能设置，原因在于单个数据点更新的wk在计算梯度的范数（或者说模长）时，
        由于其不稳定（震荡）的特性，会很快被判断到小于epsilon从而过早的停止迭代，而无法到达或者逼近最值

        震荡的原因：梯度下降的梯度计算是基于全部数据计算结果的平均，而随机梯度下降则是基于某一个数据点，因此其随机性较强
        '''
        # ew = []
        # for i in range(X.shape[1]):
        #     ewi = np.mean(-y*X[:, i]*np.exp(-y*(X@w))/(1+np.exp(-y*(X@w))))
        #     ew.append(ewi)
        # return np.array(ew)
        return -y*X*np.exp(-y*(np.dot(w,X)))/(1+np.exp(-y*(np.dot(w,X))))

    def train(self):
        i_,norm = None,None
        for i in range(self.epochs):
            idx = np.random.randint(0,self.X.shape[0])
            if self.sgd:
                drhdwk = self.sdrhd(self.X[idx],self.y[idx],self.wk)
            else:
                drhdwk = self.drhd(self.X,self.y,self.wk)
            i_,norm = i,np.linalg.norm(drhdwk)
            if norm < self.epsilon:
                break
            self.wk = self.wk-self.eta*drhdwk
        return i_,norm,self.wk

    def sign(self,value,threshold):
        return 1 if value>=threshold else -1

    def predict(self,x):
        return self.sign(self.h(np.append(np.array(1),x)),.5)

    def predict_prob(self,x):
        return self.h(np.append(np.array(1),x))

if __name__ == '__main__':
    X = np.array([[5,2], [3,2], [2,7], [1,4], [6,1], [4,5], [2,4.5]])
    y = np.array([-1, -1, 1, 1, -1, 1, -1, ])
    model = LogisticRegression(X=X,y=y,epochs=10000,eta=.2,epsilon=0.0001)
    i,norm,w = model.train()

    t1 = time.time()
    X = np.array([[5,2], [3,2], [2,7], [1,4], [6,1], [4,5], [2,4.5]])
    y = np.array([-1, -1, 1, 1, -1, 1, -1, ])
    model = LogisticRegression(X=X,y=y,epochs=5000,eta=.2,epsilon=0)
    i,norm,w = model.train()
    print(f"time={(time.time()-t1)/1000:>2f} epochs={i} -> w={w} -> norm={norm:>.8f}")
    print(y)
    print([model.predict(x) for x in X])

    t1 = time.time()
    X = np.array([[5,2], [3,2], [2,7], [1,4], [6,1], [4,5], [2,4.5]])
    y = np.array([-1, -1, 1, 1, -1, 1, -1, ])
    model = LogisticRegression(X=X,y=y,epochs=5000,eta=.2,epsilon=0,sgd=True)
    i,norm,w = model.train()
    print(f"time={(time.time()-t1)/1000:>2f} epochs={i} -> w={w} -> norm={norm:>.8f}")
    print(y)
    print([model.predict(x) for x in X])

    iris = load_iris()
    X = iris.data
    y = iris.target
    model = LogisticRegression(X=X,y=y,epochs=10000,eta=.1,epsilon=0.0005)
    i,norm,w = model.train()
    print(f"epochs={i} -> w={w} -> norm={norm:>.8f}")

    iris = load_iris()
    X = iris.data
    y = iris.target
    model = LogisticRegression(X=X,y=y,epochs=10000,eta=.1,epsilon=0,sgd=True)
    i,norm,w = model.train()
    print(f"epochs={i} -> w={w} -> norm={norm:>.8f}")
