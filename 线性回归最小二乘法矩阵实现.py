import numpy as np
import matplotlib.pyplot as plt

rnd = np.random.RandomState(3)  # 为了演示，采用固定的随机

'''
单变量线性回归最小二乘法的矩阵实现：矩阵实现的优势在于numpy本身支持伪逆；

其实就是对于误差平方和的矩阵形式对于W求导并令其为0，得到w_hat = (X^T*X)^-1*X^T*Y，其中(X^T*X)^-1*X^T称为伪逆（pseudo inverse，即函数pinv）

因此可以省略中间大量的构造经验误差、解偏导方程组等步骤；
'''

class LinearRegression(object):
    def __init__(self,X,y):
        ones = np.ones(X.shape[0]).reshape(-1,1) # 1用于计算b
        self.X = np.hstack((ones,X))
        self.y = y

    def train(self):
        # 注意，虽然一般情况下下面二者是等价的，但是在矩阵无法求逆或某些其他情况下时，二者并不相等
        # 相对而言伪逆定义更加宽泛，用处更广，因此可以的情况下建议使用伪逆
        # self.w = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        self.w = np.linalg.pinv(self.X) @ self.y
        self.w = self.w.reshape(-1)
        self.w,self.b = self.w[1:],self.w[0]
        return self.w,self.b

    def predict(self,x):
        return self.w.dot(x)+self.b

    def get(self):
        return self.X,self.y,self.w,self.b

if __name__ == '__main__':
    X0 = np.array([1.51,1.64,1.6,1.73,1.82,1.87]).reshape(-1,1)
    y = np.array([1.63,1.7,1.71,1.72,1.76,1.86])

    model = LinearRegression(X=X0,y=y)
    w,b = model.train()
    print(f'最小二乘法的矩阵方式结果为：w={w} b={b}')
    print(model.predict(np.array([X0[0]])))
    
    plt.scatter(X0,y)
    plt.plot([min(X0),max(X0)],[model.predict(min(X0)),model.predict(max(X0))])
    plt.show()

