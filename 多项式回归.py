import numpy as np
from itertools import combinations
from functools import reduce
from 线性回归最小二乘法矩阵实现 import LinearRegression as LR

class PolynomialRegression(LR):
    def __init__(self,X,y,degrees=1):
        self.combs = self.build_combs(X.shape[1],degrees)
        X = np.array([self.polynomial(x) for x in X])
        super(PolynomialRegression,self).__init__(X,y)

    def predict(self,x):
        x = self.polynomial(x)
        return super(PolynomialRegression,self).predict(x)

    def build_combs(self,elements,times):
        '''
        构建多项式的元组合
        elements 元数
        times 次数
        '''
        x_list = sum([[i]*times for i in range(elements)],[]) # 二元二次 [1 1 2 2]
        combs = sum([list(set(combinations(x_list,i))) for i in range(1,times+1)],[]) # 二元二次 [[1 1] [2 2] [1 2] [1] [2]]
        return [list(comb) for comb in combs]

    def polynomial(self,x):
        '''
        x shape = [1 N]
        '''
        fun = lambda x,y:x*y
        return [reduce(fun,x[comb]) for comb in self.combs]
