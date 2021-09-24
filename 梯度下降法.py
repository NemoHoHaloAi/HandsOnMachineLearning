import numpy as np
import matplotlib.pyplot as plt

'''
使用梯度下降法求解函数最小值，此处函数为抛物线函数y=x^2，可知最小值在x=0处，也可以通过求导得到该值；

简单说就是直接求导获取最值的方式因为公式复杂变得困难时，可以考虑以梯度的反方向为参考更新参数，以逼近最值，当然这局限于凸函数求解；
'''

xk = [10] # 初始x0
lr = .1 # 学习率η，学习率过大会导致无法收敛到最优，最小会导致逼近速度过慢
epochs = 100 # 迭代次数
e = 0.0000001 # 上限ε

def f(x):
    return x**2

def grad(x):
    return 2*x

for i in range(epochs):
    gradient_fx0 = grad(xk[i])
    if gradient_fx0 < e:
        break
    xk_1 = xk[i] - lr*gradient_fx0
    xk.append(xk_1)

print(len(xk))
print([(x,f(x)) for x in xk])

plt.plot([x*.1 for x in range(-150,150)],[f(x*.1) for x in range(-150,150)])
plt.scatter(xk,[f(x) for x in xk])
plt.show()
