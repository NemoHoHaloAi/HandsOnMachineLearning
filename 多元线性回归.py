from sympy import symbols,diff,solve
import numpy as np
from sklearn.datasets import load_boston
import timeit

X,y = load_boston(return_X_y=True)

start = timeit.default_timer()
w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,b = symbols('w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 b',real=True)
w = (w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13)

RDh = 0 # 经验误差：误差平方和
for xi,yi in zip(X,y):
    err = yi - b
    for xii,wi in zip(xi,w):
        err -= xii*wi
    RDh += err**2
RDh /= len(X)

stop = timeit.default_timer()

print(f'构造经验误差部分耗时：{stop-start:>.2f}s')
