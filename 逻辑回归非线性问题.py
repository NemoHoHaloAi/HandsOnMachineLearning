from 逻辑回归随机梯度下降 import LogisticRegression as LR
import numpy as np
import matplotlib.pyplot as plt

'''
此处解决非线性问题是通过多项式特征转换，将数据空间进行调整，调整后的空间中数据线性可分，这与之前感知机处理非线性的方法是一致的，事实上对于所有线性模型，该方法都是有效的；
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

if __name__=='__main__':
    # 数据集
    X = np.array([[-1.8, 0.6], [0.48, -1.36], [1.44, 0.52], [3.42, 3.5], [3.68, -3.64], [-4.18, 1.68]])
    Z = X**2
    y = np.array([1, 1, 1, -1, -1, -1])

    model = LR(X=X,y=y,epochs=10000,eta=.1,epsilon=0,sgd=True)
    i,norm,w = model.train()
    print(f"非线性问题 epochs={i} -> w={w} -> norm={norm:>.8f}")
    pain(141,'Before coordinate translate','x1','x2',model=model,X=X,y=y)

    model = LR(X=Z,y=y,epochs=10000,eta=.1,epsilon=0,sgd=True)
    i,norm,w = model.train()
    print(f"二阶椭圆特征转换 epochs={i} -> w={w} -> norm={norm:>.8f}")
    pain(142,'After coordinate translate','x1^2','x2^2',model=model,X=Z,y=y)
    pain(143,'Back to original coordinate','x1','x2',model=model,X=X,y=y,transform=trans_z)

    Z = np.array([[x[0]**2,x[1]**2,x[0]*x[1],x[0],x[1]] for x in X])
    model = LR(X=Z,y=y,epochs=10000,eta=.1,epsilon=0,sgd=True)
    i,norm,w = model.train()
    print(f"二阶任意曲线特征转换 epochs={i} -> w={w} -> norm={norm:>.8f}")
    pain(144,'Back to original coordinate','x1','x2',model=model,X=X,y=y,transform=trans_z2)

    plt.show()
