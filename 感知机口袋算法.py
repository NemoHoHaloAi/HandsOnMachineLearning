import numpy as np
import matplotlib.pyplot as plt

'''
感知机口袋算法：与暴力法不同，口袋算法提供了两项改动，首先通过epochs控制完整的数据训练次数，其次同步保存目前的最优w和b，这使得它可以在有限次数内处理可分和不可分情况；
'''

class Perceptron(object):
    # 初始化 w 和 b，np.array 相当于定义向量
    def __init__(self,X,y,w=np.array([0,0]),b=0,epochs=10):
        self.w,self.b = w,b
        self.X,self.y = X,y
        self.epochs = epochs

    # 定义点积函数
    def dot(self,x,w,b):
        return np.dot(w,x)+b # np.dot 是向量的点积
    
    # 定义符号函数
    def sign(self,y):
        return 1 if y>=0 else -1
    
    # 定义误差评分函数
    def score(self,X,y,w,b):
        return sum([yi*self.sign(self.dot(xi,w,b)) for xi,yi in zip(X,y)])

    def train(self):
        self.best_w,self.best_b,self.best_score = self.w,self.b,self.score(self.X,self.y,self.w,self.b)
        for _ in range(self.epochs):
            for xi,yi in zip(self.X,self.y):
                if yi*self.dot(xi,self.w,self.b)<=0:
                    self.w,self.b = self.w+yi*xi,yi+self.b
                    score_ = self.score(self.X,self.y,self.w,self.b)
                    if score_ > self.best_score:
                        self.best_w,self.best_b = self.w,self.b
                        self.best_score = score_
                    break
        return self.best_w,self.best_b

    def predict(self,x):
        return self.sign(self.dot(x,self.w,self.b))

    def get(self):
        return self.X,self.y,self.w,self.b,self.best_w,self.best_b


if __name__ == "__main__":
    # 历史信用卡发行数据，该数据不是线性可分的
    X = np.array([[5,2], [3,2], [2,7], [1,4], [6,1], [4,5], [2,4.5]])
    y = np.array([-1, -1, 1, 1, -1, 1, -1, ])

    model = Perceptron(X=X,y=y,epochs=100)
    best_w,best_b = model.train()
    print(best_w,best_b)
    
    positive = [xi for xi,yi in zip(X,y) if yi==1]
    negative = [xi for xi,yi in zip(X,y) if yi==-1]
    line = [(-best_w[0]*x-best_b)/best_w[1] for x in [-100,100]]
    plt.title('w='+str(best_w)+', b='+str(best_b))
    plt.scatter([x[0] for x in positive],[x[1] for x in positive],c='green',marker='o')
    plt.scatter([x[0] for x in negative],[x[1] for x in negative],c='red',marker='x')
    plt.plot([-100,100],line,c='black')
    plt.xlim(min([x[0] for x in X])-1,max([x[0] for x in X])+1)
    plt.ylim(min([x[1] for x in X])-1,max([x[1] for x in X])+1)
    
    plt.show()
