import numpy as np
import matplotlib.pyplot as plt

'''
感知机：线性二分类模型，拟合分割超平面对数据进行分类；
暴力实现：无脑针对每一个错误点进行w和b的更新，可以证明在线性可分情况下，有限次迭代可以完成划分；
'''

# 初始化 w 和 b，np.array 相当于定义向量
w,b = np.array([0, 0]),0 

# 定义 d(x) 函数
def d(x):
    return np.dot(w,x)+b # np.dot 是向量的点积

# 历史信用卡发行数据
# 这里的数据集不能随便修改，否则下面的暴力实现可能停不下来
X = np.array([[5,2], [3,2], [2,7], [1,4], [6,1], [4,5]])
Y = np.array([-1, -1, 1, 1, -1, 1])

run = True
while run:
    run = False
    for x,y in zip(X,Y):
        if y*d(x)<=0:
            w,b = w+y*x,y+b
            run = True
            break

print(w,b)

positive = [x for x,y in zip(X,Y) if y==1]
negative = [x for x,y in zip(X,Y) if y==-1]
line = [(-w[0]*x-b)/w[1] for x in [-100,100]]
plt.title('w='+str(w)+', b='+str(b))
plt.scatter([x[0] for x in positive],[x[1] for x in positive],c='green',marker='o')
plt.scatter([x[0] for x in negative],[x[1] for x in negative],c='red',marker='x')
plt.plot([-100,100],line,c='black')
plt.xlim(min([x[0] for x in X])-1,max([x[0] for x in X])+1)
plt.ylim(min([x[1] for x in X])-1,max([x[1] for x in X])+1)

plt.show()
