import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split as tts
from 多项式回归 import PolynomialRegression as PR

rnd = np.random.RandomState(3)
x_min, x_max = 0, 10

def pain(pos=141,xlabel='x',ylabel='y',title='',x=[],y=[],line_x=[],line_y=[]):
    plt.subplot(pos)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x,y)
    plt.plot(line_x,line_y)

# 上帝函数 y=f(x)
def f(x):
    return x**5-22*x**4+161*x**3-403*x**2+36*x+938

# 上帝分布 P(Y|X)
def P(X):
    return f(X) + rnd.normal(scale=30, size=X.shape)

# 通过 P(X, Y) 生成数据集 D
X = rnd.uniform(x_min, x_max, 50)   # 通过均匀分布产生 X
y = P(X)                            # 通过 P(Y|X) 产生 y

plt.subplot(332)
plt.scatter(x=X, y=y)
xx = np.linspace(x_min, x_max)
plt.plot(xx, f(xx), 'k--')

X_train,X_test,y_train,y_test = tts(X,y,test_size=0.3,random_state=10086)
X_train,X_test,y_train,y_test = X_train.reshape(-1,1),X_test.reshape(-1,1),y_train.reshape(-1,1),y_test.reshape(-1,1)

for pos,deg in zip([334,335,336,337,338,339],[1,3,5,8,15,20]):
    model = PR(X=X_train,y=y_train,degrees=deg)
    w,b = model.train()
    x_min,x_max = min(X_train),max(X_train)
    line_x = [x_min+(x_max-x_min)*(i/100) for i in range(100)]
    line_y = [model.predict(x) for x in line_x]
    pain(pos,'x','y','DEG='+str(deg),X_train[:,0],y_train[:,0],line_x,line_y)

plt.tight_layout()
plt.show()

for pos,deg in zip([334,335,336,337,338,339],[1,3,5,8,15,20]):
    model = PR(X=X_train,y=y_train,degrees=deg)
    w,b = model.train()
    print(w,b)
    x_min,x_max = min(X_test),max(X_test)
    line_x = [x_min+(x_max-x_min)*(i/100) for i in range(100)]
    line_y = [model.predict(x) for x in line_x]
    mse = sum([((model.predict(x)-y)**2)[0] for x,y in zip(X_test,y_test)])/len(X_test)
    pain(pos,'x','y','DEG='+str(deg)+',MSE='+str(int(mse)),X_test[:,0],y_test[:,0],line_x,line_y)

plt.tight_layout()
plt.show()
