import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from 线性回归最小二乘法矩阵实现 import LinearRegression as LR


'''
单变量线性回归最小二乘法的矩阵实现：矩阵实现的优势在于numpy本身支持伪逆；

其实就是对于误差平方和的矩阵形式对于W求导并令其为0，得到w_hat = (X^T*X)^-1*X^T*Y，其中(X^T*X)^-1*X^T称为伪逆（pseudo inverse，即函数pinv）

因此可以省略中间大量的构造经验误差、解偏导方程组等步骤；
'''

if __name__ == '__main__':
    X0,y = load_boston(return_X_y=True)
    model = LR(X=X0,y=y)
    w,b = model.train()
    print(f'多元线性回归最小二乘法的矩阵方式结果为：w={w} b={b}')

    X0 = X0[:,::-1]
    model = LR(X=X0,y=y)
    w,b = model.train()
    print(f'多元线性回归最小二乘法的矩阵方式结果为：w={w} b={b}')

    X0 = X0[:2,:]
    y = y[:2]
    model = LR(X=X0,y=y)
    w,b = model.train()
    print(f'多元线性回归最小二乘法的矩阵方式结果为：w={w} b={b}')

    X0 = X0[:,::-1]
    model = LR(X=X0,y=y)
    w,b = model.train()
    print(f'多元线性回归最小二乘法的矩阵方式结果为：w={w} b={b}')
