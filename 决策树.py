import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def pain(pos=121,title='',xlabel='',ylabel='',resolution=0.05,model=None,X=[],y=[],line_x=[],line_y=[],transform=None):
    plt.subplot(pos)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    xy_min = min(min([x[0] for x in X]),min([x[1] for x in X]))
    xy_max = max(max([x[0] for x in X]),max([x[1] for x in X]))
    print(xy_min,xy_max)
    xx1, xx2 = np.mgrid[xy_min-1:xy_max+1.1:resolution, xy_min-1:xy_max+1.1:resolution]
    print(xx1,xx2)
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    print(grid)
    if transform:
        grid = transform(grid)
    y_pred = np.array([model.predict([x]) for x in grid]).reshape(xx1.shape)
    print(y_pred)
    plt.contourf(xx1, xx2, y_pred, 25, cmap="coolwarm", vmin=0, vmax=1, alpha=0.8)

    plt.scatter([xi[0] for xi,yi in zip(X,y) if yi==1],[xi[1] for xi,yi in zip(X,y) if yi==1],c='black',marker='o')
    plt.scatter([xi[0] for xi,yi in zip(X,y) if yi==-1],[xi[1] for xi,yi in zip(X,y) if yi==-1],c='black',marker='x')

class DecisionTree(object):
    def __init__(self,criterion="CART",max_depth=None,min_samples_split=2):
        '''
        criterion : 节点分裂依据指标，默认CART，可选gini-CART、gain-ID3、gain_ratio-C4.5
        max_depth : 树最大深度，默认没有限制
        min_samples_split : 节点最小分裂所需样本量，默认2
        '''
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def p(self,y):
        return {k:len(y[y==k])/len(y) for k in list(set(y))}

    def gini(self,y):
        P = self.p(y)
        return 1 - sum([P[k]**2 for k in list(set(y))])

    def gini_index(self,X,i,y):
        '''
        统一修正为值越大越好：负号
        '''
        v = list(set(X[:,i]))
        m,n = X.shape
        return sum([-len(X[X[:,i]==vv])/m*self.gini(y[X[:,i]==vv]) for vv in v])

    def entropy(self,y):
        '''
        信息熵
        '''
        P = self.p(y)
        return -sum([P[yy]*np.log2(P[yy]) for yy in np.unique(y)])

    def gain(self,X,i,y):
        '''
        统一修正为值越大越好
        '''
        v = list(set(X[:,i]))
        m,n = X.shape
        return self.entropy(y) - sum([len(y[X[:,i]==vv])/m*self.entropy(y[X[:,i]==vv]) for vv in v])

    def gain_ratio(self,X,i,y):
        '''
        统一修正为值越大越好
        '''
        v = list(set(X[:,i]))
        m,n = X.shape
        iv = - sum([(len(X[X[:,i]==vv])/m)*np.log2(len(X[X[:,i]==vv])/m) for vv in v])
        iv = iv+1e-10 if iv==0 else iv # 避免divide0异常
        return self.gain(X,i,y)/iv

    def method(self,X,i,y):
        if self.criterion == "CART":
            return self.gini_index(X,i,y)
        if self.criterion == "ID3":
            return self.gain(X,i,y)
        if self.criterion == "C4.5":
            return self.gain_ratio(X,i,y)
        return self.gini_index(X,i,y)

    def split_node(self,node,X,idx,y):
        for v in list(set(X[:,idx])):
            node["sub_nodes"]["col"+str(idx)+"="+str(v)] = {"depth":node["depth"]+1,"X":X[X[:,idx]==v],"y":y[X[:,idx]==v],"split_idx":None,"v":v,"sub_nodes":{}}
        for sub_name,sub_node in node["sub_nodes"].items():
            self.split(sub_node)

    def split(self,node):
        X,y = node["X"],node["y"]
        m,n = X.shape
        if m >= 2 and n > 1:
            ginis = [self.method(X,i,y) for i in range(n)]
            idx = np.argmax(ginis)
            node["split_idx"] = idx
            self.split_node(node,X,idx,y)

    def train(self,X,y):
        self.root = {"depth":0,"X":X,"y":y,"split_idx":None,"v":None,"sub_nodes":{}}
        self.split(self.root)

    def print_node(self,name,node):
        if node:
            print("\t"*node["depth"]+name+":split "+str(node["split_idx"]))
            if node["sub_nodes"]:
                for sub_name,sub_node in node["sub_nodes"].items():
                    self.print_node(sub_name,sub_node)

    def print(self):
        self.print_node("root",self.root)

    def go(self,node,x):
        if len(node["sub_nodes"]) <= 0:
            return node["y"]
        for sub_name,sub_node in node["sub_nodes"].items():
            if x[node["split_idx"]] == sub_node["v"]:
                return self.go(sub_node,x)

    def predict(self,x):
        print(x)
        result = self.go(self.root,x)
        max_count,max_y = 0,0
        for yy in list(set(result)):
            if len([yi for yi in y if yi==yy]) > max_count:
                max_count,max_y = len([yi for yi in y if yi==yy]),yy
        return yy


if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [2, 1]])
    y = np.array([1, -1, -1, 1, 1, -1,])
    print(X)
    print(y)

    criterion = "CART"
    print(f"Decision Tree: criterion={criterion}")
    tree = DecisionTree(criterion=criterion)
    tree.train(X,y)
    tree.print()
    for xi,yi in zip(X,y):
        print("pred:",tree.predict(xi),", real:",yi)
    print("------------------------------------------------------")

    criterion = "ID3"
    print(f"Decision Tree: criterion={criterion}")
    tree = DecisionTree(criterion=criterion)
    tree.train(X,y)
    tree.print()
    for xi,yi in zip(X,y):
        print("pred:",tree.predict(xi),", real:",yi)
    print("------------------------------------------------------")

    criterion = "C4.5"
    print(f"Decision Tree: criterion={criterion}")
    tree = DecisionTree(criterion=criterion)
    tree.train(X,y)
    tree.print()
    for xi,yi in zip(X,y):
        print("pred:",tree.predict(xi),", real:",yi)
    print("------------------------------------------------------")
