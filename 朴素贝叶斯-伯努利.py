import numpy as np

'''
生成式模型：假设数据X由正类分布Z1和负类分布Z2生成，通过现有数据X计算出分布Z1、Z2的参数后（比如正态分布的μ和的σ^2，即均值和方差），最后通过两个分布来进行预测推断；

伯努利：此处X字段均为离散字段，因此可以使用伯努利分布来表示其生成分布；

贝叶斯：
    预测：P(Y=+1|X1=1,X2=1) 即点(1,1)属于正类的概率，后验概率；
    根据贝叶斯公式有：
        P(Y=+1|X1=1,X2=1) = ( P(Y=+1) * P(X1=1,X2=1|Y=+1) ) / (P(X1=1,X2=1))
        P(Y=+1) 正类的数据比例
        P(X1=1,X2=1) (1,1)的数据比例
        P(X1=1,X2=1|Y=+1) 前验概率，即正类中(1,1)的比例

朴素贝叶斯：
    原始贝叶斯的前验概率P(X1=1,X2=1|Y=+1)，假设Y∈{+1,=1}，X为2维向量，每个维度只有0/1两个取值
    那么需要计算的前验概率有：2个类别 * 2个特征^2个取值 = 8，假设每个组合需要数据量为100，那么当前二维二分类问题需要800个数据点

    朴素：假设各个维度条件独立，数学表达为：P(X1=1,X2=1|Y=+1) = P(X1=1|Y=+1) * P(X2=1|Y=+1)
    因此需要计算的概率有：2个类别 * 2个特征*2个取值 = 8

    以上可以看到，计算的概率数量以及需要的数据量正比从指数2^n降低为多项式2*n，即在高维情况下使得数据量的需求依然可以满足；
'''

class Bayes(object):
    def __init__(self,X,y,naive=False):
        self.X = X
        self.y = y
        self.naive = naive

    def train(self):
        # 前验概率 P(Y=?)
        self.p_positive = 1.*len([yi for yi in y if yi==1])/len(y)
        self.p_negative = 1 - self.p_positive

        # 似然率 P(X1=?,X2=?)
        self.likelihood = [[0 for i in range(2)] for j in range(2)]
        self.likelihood[0][0] = 1.*len([x for x in X if x[0]==0 and x[1]==0])/len(X)
        self.likelihood[0][1] = 1.*len([x for x in X if x[0]==0 and x[1]==1])/len(X)
        self.likelihood[1][0] = 1.*len([x for x in X if x[0]==1 and x[1]==0])/len(X)
        self.likelihood[1][1] = 1.*len([x for x in X if x[0]==1 and x[1]==1])/len(X)

        # P(X1=?,X2=?|Y=?)
        X_p,X_n = X[y==1],X[y==-1]
        self.pre_probability = [[[0 for i in range(2)] for j in range(2)] for k in range(2)]
        if self.naive: # 朴素方式计算
            p_x1_0_y_0 = 1.*len([x for x in X_p if x[0]==0])/len(X_p)
            p_x1_1_y_0 = 1.*len([x for x in X_p if x[0]==1])/len(X_p)
            p_x2_0_y_0 = 1.*len([x for x in X_p if x[1]==0])/len(X_p)
            p_x2_1_y_0 = 1.*len([x for x in X_p if x[1]==1])/len(X_p)
            p_x1_0_y_1 = 1.*len([x for x in X_n if x[0]==0])/len(X_n)
            p_x1_1_y_1 = 1.*len([x for x in X_n if x[0]==1])/len(X_n)
            p_x2_0_y_1 = 1.*len([x for x in X_n if x[1]==0])/len(X_n)
            p_x2_1_y_1 = 1.*len([x for x in X_n if x[1]==1])/len(X_n)
            self.pre_probability[0][0][0] = p_x1_0_y_0 * p_x2_0_y_0 
            self.pre_probability[0][0][1] = p_x1_0_y_0 * p_x2_1_y_0 
            self.pre_probability[0][1][0] = p_x1_1_y_0 * p_x2_0_y_0 
            self.pre_probability[0][1][1] = p_x1_1_y_0 * p_x2_1_y_0 
            self.pre_probability[1][0][0] = p_x1_0_y_1 * p_x2_0_y_1 
            self.pre_probability[1][0][1] = p_x1_0_y_1 * p_x2_1_y_1 
            self.pre_probability[1][1][0] = p_x1_1_y_1 * p_x2_0_y_1 
            self.pre_probability[1][1][1] = p_x1_1_y_1 * p_x2_1_y_1 
        else:
            self.pre_probability[0][0][0] = 1.*len([x for x in X_p if x[0]==0 and x[1]==0])/len(X_p)
            self.pre_probability[0][0][1] = 1.*len([x for x in X_p if x[0]==0 and x[1]==1])/len(X_p)
            self.pre_probability[0][1][0] = 1.*len([x for x in X_p if x[0]==1 and x[1]==0])/len(X_p)
            self.pre_probability[0][1][1] = 1.*len([x for x in X_p if x[0]==1 and x[1]==1])/len(X_p)
            self.pre_probability[1][0][0] = 1.*len([x for x in X_n if x[0]==0 and x[1]==0])/len(X_n)
            self.pre_probability[1][0][1] = 1.*len([x for x in X_n if x[0]==0 and x[1]==1])/len(X_n)
            self.pre_probability[1][1][0] = 1.*len([x for x in X_n if x[0]==1 and x[1]==0])/len(X_n)
            self.pre_probability[1][1][1] = 1.*len([x for x in X_n if x[0]==1 and x[1]==1])/len(X_n)

    def predict(self,x):
        positive = self.p_positive * self.pre_probability[0][x[0]][x[1]]
        negative = self.p_negative * self.pre_probability[1][x[0]][x[1]]
        return positive,negative,1 if positive>negative else -1


if __name__ == "__main__":
    X = np.array([[1,1],[1,1],[1,0],[0,1],[1,0],[0,1],[0,1],[0,0],[0,0],[1,1]])
    y = np.array([-1,-1,-1,-1,1,1,1,1,1,1])
    print(X)
    print(y)

    bayes = Bayes(X,y)
    bayes.train()
    print(f'data={X[0]} bayes\tpredict={bayes.predict(X[0])}')

    bayes = Bayes(X,y,naive=True)
    bayes.train()
    print(f'data={X[0]} naive bayes\tpredict={bayes.predict(X[0])}')
