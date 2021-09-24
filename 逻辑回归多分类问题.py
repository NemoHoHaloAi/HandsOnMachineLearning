from 逻辑回归随机梯度下降 import LogisticRegression as LR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

'''
基于 one-vs-rest、one-vs-one等方式结合二分类模型实现多分类效果；
'''

# def pain(pos=121,title='',xlabel='',ylabel='',resolution=0.05,model=None,X=[],y=[],line_x=[],line_y=[],transform=None):
#     plt.subplot(pos)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
# 
#     xy_min = min(min([x[0] for x in X]),min([x[1] for x in X]))
#     xy_max = max(max([x[0] for x in X]),max([x[1] for x in X]))
#     xx1, xx2 = np.mgrid[xy_min-1:xy_max+1.1:resolution, xy_min-1:xy_max+1.1:resolution]
#     grid = np.c_[xx1.ravel(), xx2.ravel()]
#     if transform:
#         grid = transform(grid)
#     y_pred = np.array([model.predict(np.array(x)) for x in grid]).reshape(xx1.shape)
#     plt.contourf(xx1, xx2, y_pred, 25, cmap="coolwarm", vmin=0, vmax=1, alpha=0.8)
# 
#     plt.scatter([xi[0] for xi,yi in zip(X,y) if yi==1],[xi[1] for xi,yi in zip(X,y) if yi==1],c='black',marker='o')
#     plt.scatter([xi[0] for xi,yi in zip(X,y) if yi==-1],[xi[1] for xi,yi in zip(X,y) if yi==-1],c='black',marker='x')
#     # plt.plot(line_x,line_y,color='black')

if __name__=='__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    # one vs rest
    y1 = y.copy()
    y1[y1!=0]=-1
    y1[y1==0]=1
    model1 = LR(X=X,y=y1,epochs=10000,eta=.1,epsilon=0.00001)
    i,norm,w = model1.train()
    print(f"one-vs-rest\tepochs={i} -> w={w} -> norm={norm:>.8f}")

    y2 = y.copy()
    y2[y2!=1]=-1
    y2[y2==1]=1
    model2 = LR(X=X,y=y2,epochs=10000,eta=.1,epsilon=0.00001)
    i,norm,w = model2.train()
    print(f"one-vs-rest\tepochs={i} -> w={w} -> norm={norm:>.8f}")

    y3 = y.copy()
    y3[y3!=2]=-1
    y3[y3==2]=1
    model3 = LR(X=X,y=y3,epochs=10000,eta=.1,epsilon=00.00001)
    i,norm,w = model3.train()
    print(f"one-vs-rest\tepochs={i} -> w={w} -> norm={norm:>.8f}")

    # one vs one
    X1 = X[y!=2] # 0 1
    y1 = y[y!=2]
    y1[y1==1]=-1
    y1[y1==0]=1
    model1_ = LR(X=X1,y=y1,epochs=10000,eta=.1,epsilon=00.00001)
    i,norm,w = model1_.train()
    print(f"one-vs-one\tepochs={i} -> w={w} -> norm={norm:>.8f}")

    X2 = X[y!=0] # 1 2
    y2 = y[y!=0]
    y2[y2==2]=-1
    y2[y2==1]=1
    model2_ = LR(X=X2,y=y2,epochs=10000,eta=.1,epsilon=00.00001)
    i,norm,w = model2_.train()
    print(f"one-vs-one\tepochs={i} -> w={w} -> norm={norm:>.8f}")
    
    X3 = X[y!=1] # 0 2
    y3 = y[y!=1]
    y3[y3==0]=-1
    y3[y3==2]=1
    model3_ = LR(X=X3,y=y3,epochs=10000,eta=.1,epsilon=00.00001)
    i,norm,w = model3_.train()
    print(f"one-vs-one\tepochs={i} -> w={w} -> norm={norm:>.8f}")

    correct_ovr,correct_ovo = 0,0

    plt.subplot(121)
    for xi,yi in zip(X,y):
        pred1,pred2,pred3 = model1.predict_prob(np.array(xi)),model2.predict_prob(np.array(xi)),model3.predict_prob(np.array(xi))
        pred = 0 if max([pred1,pred2,pred3])==pred1 else(1 if max([pred1,pred2,pred3])==pred2 else 2)
        correct_ovr += 1 if pred==yi else 0
        plt.scatter(xi[2],xi[3],marker='o' if pred==0 else ('x' if pred==1 else '^'),s=30,c='green' if pred==yi else 'red')
    plt.title(f'one vs rest model correct rate:{correct_ovr/X.shape[0]*100:>.2f}%')

    plt.subplot(122)
    for xi,yi in zip(X,y):
        pred1_,pred2_,pred3_ = model1_.predict(np.array(xi)),model2_.predict(np.array(xi)),model3_.predict(np.array(xi))
        pred1_ = 0 if pred1_==1 else 1
        pred2_ = 1 if pred2_==1 else 2 
        pred3_ = 2 if pred3_==1 else 0 
        count_list = [[pred1_,pred2_,pred3_].count(0),[pred1_,pred2_,pred3_].count(1),[pred1_,pred2_,pred3_].count(2)]
        pred_ = 0 if max(count_list)==count_list[0] else(1 if max(count_list)==count_list[1] else 2)
        correct_ovo += 1 if pred_==yi else 0
        plt.scatter(xi[2],xi[3],marker='o' if pred_==0 else ('x' if pred_==1 else '^'),s=30,c='green' if pred_==yi else 'red')
    plt.title(f'one vs rest model correct rate:{correct_ovo/X.shape[0]*100:>.2f}%')

    plt.show()
