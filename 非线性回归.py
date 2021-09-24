import numpy as np
import matplotlib.pyplot as plt
from 线性回归最小二乘法矩阵实现 import LinearRegression as LR
from 多项式回归 import PolynomialRegression as PR

plt.figure(figsize=(18,4))

def pain(pos=141,xlabel='x',ylabel='y',title='',x=[],y=[],line_x=[],line_y=[]):
    plt.subplot(pos)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x,y)
    plt.plot(line_x,line_y)

rnd = np.random.RandomState(3)  # 为了演示，采用固定的随机

X = np.array([-1+(1-(-1))*(i/10) for i in range(10)]).reshape(-1,1)
y = (X**2)+rnd.normal(scale=.1,size=X.shape)
model = LR(X=X,y=y)
w,b = model.train()
print(w,b)
line_x = [min(X[:,0]),max(X[:,0])]
line_y = [model.predict(np.array([min(X[:,0])])),model.predict(np.array([max(X[:,0])]))]
pain(151,'x','y','degress=1',X[:,0],y[:,0],line_x,line_y)

X2 = X**2
model = LR(X=X2,y=y)
w,b = model.train()
print(w,b)
line_x = [min(X2[:,0]),max(X2[:,0])]
line_y = [model.predict(np.array([x**2])) for x in line_x]
pain(152,'x^2','y','translate coord & degress=2',X2[:,0],y[:,0],line_x,line_y)

x_min,x_max = min(X[:,0]),max(X[:,0])
line_x = [x_min+(x_max-x_min)*(i/100) for i in range(100)]
line_y = [model.predict(np.array([x**2])) for x in line_x]
pain(153,'x','y','degress=2',X[:,0],y[:,0],line_x,line_y)

model = PR(X=X,y=y,degress=5)
w,b = model.train()
print(w,b)
x_min,x_max = min(X),max(X)
line_x = [x_min+(x_max-x_min)*(i/100) for i in range(100)]
line_y = [model.predict(x) for x in line_x]
pain(154,'x','y','degress=5',X[:,0],y[:,0],line_x,line_y)

model = PR(X=X,y=y,degress=10)
w,b = model.train()
print(w,b)
x_min,x_max = min(X),max(X)
line_x = [x_min+(x_max-x_min)*(i/100) for i in range(100)]
line_y = [sum(model.predict(x)) for x in line_x]
pain(155,'x','y','degress=10',X[:,0],y[:,0],line_x,line_y)

plt.tight_layout()
plt.show()

for pos,deg in zip([331,332,333,334,335,336,337,338,339],[1,3,5,7,9,11,13,15,20]):
    model = PR(X=X,y=y,degress=deg)
    w,b = model.train()
    x_min,x_max = min(X),max(X)
    line_x = [x_min+(x_max-x_min)*(i/100) for i in range(-1,101,1)]
    line_y = [sum(model.predict(x)) for x in line_x]
    pain(pos,'x','y','degress='+str(deg),X[:,0],y[:,0],line_x,line_y)
plt.tight_layout()
plt.show()
