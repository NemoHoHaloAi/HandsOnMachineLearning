from sympy import symbols, diff, solve
import numpy as np
import matplotlib.pyplot as plt

'''
线性回归拟合wx+b直线；

最小二乘法指的是优化求解过程是通过对经验误差（此处是均平方误差）求偏导并令其为0以解的w和b；
'''

# 数据集 D X为父亲身高，Y为儿子身高
X = np.array([1.51, 1.64, 1.6, 1.73, 1.82, 1.87])
y = np.array([1.63, 1.7, 1.71, 1.72, 1.76, 1.86])

# 构造符号
w,b = symbols('w b',real=True)

# 定义经验误差计算公式：(1/N)*sum(yi-(w*xi+b))^2)
RDh = 0
for xi,yi in zip(X,y):
    RDh += (yi - (w*xi+b))**2
RDh = RDh / len(X)

# 对w和b求偏导：求偏导的结果是得到两个结果为0的方程式
eRDHw = diff(RDh,w)
eRDHb = diff(RDh,b)

# 求解联立方程组
ans = solve((eRDHw,eRDHb),(w,b))
w,b = ans[w],ans[b]
print('使得经验误差RDh取得最小值的参数为：'+str(ans))

plt.scatter(X,y)
x_range = [min(X)-0.1,max(X)+0.1]
y_range = [w*x_range[0]+b,w*x_range[1]+b]
plt.plot(x_range,y_range)

plt.show()
