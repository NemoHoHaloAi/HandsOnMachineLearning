import numpy as np
import matplotlib.pyplot as plt
from 感知机口袋算法 import Perceptron

plt.figure(figsize=(18,6))

'''
通过坐标转换/特征转换将非线性问题转为线性问题，再使用线性模型解决；
'''

def trans_z(X):
    return X**2

def trans_z2(X):
    return np.array([[x[0]**2,x[1]**2,x[0]*x[1],x[0],x[1]] for x in X])

def pain(pos=121,title='',xlabel='',ylabel='',resolution=0.05,model=None,X=[],y=[],line_x=[],line_y=[],transform=None):
    plt.subplot(pos)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    xy_min = min(min([x[0] for x in X]),min([x[1] for x in X]))
    xy_max = max(max([x[0] for x in X]),max([x[1] for x in X]))
    xx1, xx2 = np.mgrid[xy_min-1:xy_max+1.1:resolution, xy_min-1:xy_max+1.1:resolution]
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    if transform:
        grid = transform(grid)
    y_pred = np.array([model.predict(np.array(x)) for x in grid]).reshape(xx1.shape)
    plt.contourf(xx1, xx2, y_pred, 25, cmap="coolwarm", vmin=0, vmax=1, alpha=0.8)

    plt.scatter([xi[0] for xi,yi in zip(X,y) if yi==1],[xi[1] for xi,yi in zip(X,y) if yi==1],c='black',marker='o')
    plt.scatter([xi[0] for xi,yi in zip(X,y) if yi==-1],[xi[1] for xi,yi in zip(X,y) if yi==-1],c='black',marker='x')
    # plt.plot(line_x,line_y,color='black')

## 不可分
X = np.array([[-1.8,0.6],[0.48,-1.36],[3.68,-3.64],[1.44,0.52],[3.42,3.5],[-4.18,1.68]])
y = np.array([1,1,-1,1,-1,-1])
model = Perceptron(X=X,y=y,epochs=100)
w,b = model.train()
# 注意绘制分割直线公式为：wx+b=0，因此给定x[0]，计算对应的x[1]即可画图
# w[0]*x[0]+w[1]*x[1]+b=0 => x[1]=(-b-w[0]*x[0])/w[1]
line_x = [min([x[0] for x in X])-3,max([x[0] for x in X])+3]
line_y = [(-b-w[0]*line_x[0])/w[1],(-b-w[0]*line_x[1])/w[1]]
pain(141,'Before coordinate translate','x1','x2',model=model,X=X,y=y)

## 转换坐标为可分
Z = X**2 # z1=x1^2,z2=x2^2，相当于对原数据空间做坐标系转换，也可以理解为特征转换
model = Perceptron(X=Z,y=y,epochs=100)
w,b = model.train()
line_x = [min([x[0] for x in Z])-3,max([x[0] for x in Z])+3]
line_y = [(-b-w[0]*line_x[0])/w[1],(-b-w[0]*line_x[1])/w[1]]
pain(142,'After coordinate translate','z1=x1^2','z2=x2^2',model=model,X=Z,y=y)

## 转换回原坐标绘制分割线，此时为曲线
line_x = [(line_x[0]+(line_x[1]-line_x[0])*(i/100))**.5 for i in range(0,100)]
line_y = [((-b-w[0]*(x**2))/w[1])**.5 for x in line_x]
pain(143,'Back to original coordinate','x1','x2',model=model,X=X,y=y,transform=trans_z)

## 使用任意二次曲线转换坐标：所有可能的二元二次方程
Z2 = np.array([[x[0]**2,x[1]**2,x[0]*x[1],x[0],x[1]] for x in X])
model = Perceptron(X=Z2,y=y,w=np.array([0,0,0,0,0]),epochs=100)
w,b = model.train()
# w0*x0^2+w1*x1^2+w2*x0*x1+w3*x0+w4*x1+b=0 => x1=-b-w3*x0-w0*x0^2
# line_x = [(line_x[0]+(line_x[1]-line_x[0])*(i/100))**.5 for i in range(0,100)]
# line_y = [((-b-w[0]*(x**2))/w[1])**.5 for x in line_x]
pain(144,'Back to original coordinate','x1','x2',model=model,X=X,y=y,transform=trans_z2)

plt.show()
