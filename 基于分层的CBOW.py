import numpy as np
import random,string
import sys
sys.setrecursionlimit(1000000)

import jieba
import jieba.analyse
import jieba.posseg as pseg

jieba.enable_paddle()
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('田国富', True)
jieba.suggest_freq('高育良', True)
jieba.suggest_freq('侯亮平', True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)

punc = list(string.punctuation)+list("·~！￥……（）——、|【】；：‘“，《。》？     \r\n\t")

'''
数据：警情词频数据
'''

class HierarchicalSoftmaxCBOW(object):
    '''
    基于分层Softmax的CBOW；
    1. 输入层：词向量求和后求均值；
    2. 霍夫曼树进行迭代更新参数节点权值Θ和节点向量X_w；
    3. 预测x；
    '''

    def __init__(self,C=2,ETA=.1,SIZE=100,document="../data/人民的名义.txt"):
        '''
        C : CBOW采样法下输出词前后各取c个词
        ETA : η，即学习率
        SIZE : 词向量的维度
        document : 训练使用文本文件路径
        '''
        print("init model")
        self.C = C
        self.ETA = ETA
        self.SIZE = SIZE
        print("read document")
        self.lines = open(document).readlines()
        self.words = sum([[w for w,f in pseg.cut(line) if f not in ["m","w","d","p","vd","xc"] and w not in punc] for line in self.lines],[])
        # self.words = sum([[w.lower() for w in line.split(" ") if w not in punc] for line in self.lines],[])
        self.unique_words = list(set(self.words))
        tooless_words = [w for w in self.unique_words if self.words.count(w)<=10] # 过滤出行次数过少的词
        self.words = [w for w in self.words if w not in tooless_words]
        self.unique_words = list(set(self.words))
        self.dict_size = len(self.unique_words)
        self.word_freq = {w:self.words.count(w) for w in self.unique_words}
        print("init word vector")
        # self.xw = np.random.random((self.dict_size,self.dict_size)) # 词向量
        self.xw = np.eye(self.dict_size) # 词向量
        # self.xw = np.random.random((self.dict_size,self.SIZE)) # 词向量

    def sample(self):
        self.samples = []
        for i in range(self.C,len(self.words)-self.C-1):
            self.samples.append(self.words[i-self.C:i+self.C+1])
        self.sample_idxs = []
        for sample in self.samples:
            self.sample_idxs.append([self.unique_words.index(word) for word in sample])

    def train(self):
        print("train")
        self.sample()
        for idxs in self.sample_idxs:
            in_vector = np.array([self.xw[i] for i in idxs[:self.C]+idxs[self.C+1:]])
            out_vector = self.xw[idxs[self.C]].reshape((1,self.dict_size))
            e,mean_vector = 0,np.mean(in_vector,axis=0).reshape((1,self.dict_size))
            # out_vector = self.xw[idxs[self.C]].reshape((1,self.SIZE))
            # e,mean_vector = 0,np.mean(in_vector,axis=0).reshape((1,self.SIZE))
            node = self.tree
            while(node):
                f = 1/(1+np.exp(-np.dot(mean_vector,node["theta"].T)[0][0])) # dwj=0
                f_inv = 1-f # dwj=1
                dwj = 0 if f>f_inv else 1
                g = self.ETA*(1-dwj-f)
                e = e+g*node["theta"]
                node["theta"] = node["theta"]+g*mean_vector
                node = node["left_subnode"] if ("left_subnode" in node.keys() and dwj==1) else (node["right_subnode"] if ("right_subnode" in node.keys() and dwj==0) else None)
            for k in idxs[:self.C]+idxs[self.C+1:]:
                self.xw[k] = self.xw[k]+e.reshape((self.dict_size,))
                # self.xw[k] = self.xw[k]+e.reshape((self.SIZE,))

    def similarity(self,w1,w2):
        '''
        w1 : word1
        w2 : word2
        return similarity，cos(v1,v2)
        '''
        i1,i2 = self.unique_words.index(w1),self.unique_words.index(w2)
        if i1!=-1 and i2!=-1:
            v1,v2 = self.xw[i1],self.xw[i2]
            return float(np.dot(v1.T,v2))/(np.linalg.norm(v1)*np.linalg.norm(v2))
        return None

    def search(self,w,length=None,size=10):
        result = [(word,self.similarity(w,word)) for word in self.unique_words if length==None or len(word)==length]
        result.sort(key=lambda x:x[1],reverse=True)
        return result[:size]

    def no_match(self,names):
        idxs = [self.unique_words.index(name) for name in names]
        w,sim = "",1
        for i in range(len(names)):
            sim_tmp = sum([self.similarity(names[i],names[j]) for j in range(len(names)) if i!=j])/(len(names)-1)
            if sim_tmp < sim:
                w,sim = names[i],sim_tmp
        return w,sim

    def huffman_tree(self,nodes):
        '''
        nodes : node={weight:12,name:"我"}
        '''
        if len(nodes) > 1:
            min_idx = np.argmin([n["weight"] for n in nodes])
            second_min_idx = np.argmin([n["weight"] if i!=min_idx else 9999999999 for i,n in enumerate(nodes)])
            new_nodes = [n for i,n in enumerate(nodes) if i not in [min_idx,second_min_idx]]
            node_i,node_j = nodes[min_idx],nodes[second_min_idx]
            node = {"weight":node_i["weight"]+node_j["weight"],"name":node_i["name"]+"+"+node_j["name"],"left_subnode":node_j,"right_subnode":node_i,"theta":np.ones((1,self.dict_size))}
            # node = {"weight":node_i["weight"]+node_j["weight"],"name":node_i["name"]+"+"+node_j["name"],"left_subnode":node_j,"right_subnode":node_i,"theta":np.random.random((1,self.SIZE))}
            return self.huffman_tree(new_nodes+[node])
        return nodes

    def build_huffman_tree(self):
        '''
        1. 分词处理；
        2. 统计词频；
        3. 词频构建霍夫曼树；
        4. 根据CBOW方式采样；
        '''
        self.nodes = [{"weight":c,"name":w,"theta":np.ones((1,self.dict_size))} for w,c in self.word_freq.items()]
        # self.nodes = [{"weight":c,"name":w,"theta":np.ones((1,self.SIZE))} for w,c in self.word_freq.items()]
        self.tree = self.huffman_tree(self.nodes)[0]

    def print_tree(self,tree,t):
        if tree:
            print("\t"*t,tree["name"]+":"+str(tree["weight"]))
            if "left_subnode" in tree.keys():
                self.print_tree(tree["left_subnode"],t+1)
            if "right_subnode" in tree.keys():
                self.print_tree(tree["right_subnode"],t+1)

    def print_huffman_tree(self):
        self.print_tree(self.tree,1)


if __name__ == "__main__":
    # model = HierarchicalSoftmaxCBOW(C=3,ETA=.3,document="../data/哈利波特.txt")
    # model.build_huffman_tree()
    # model.train()
    # w1,w2 = "harry","ron"
    # print(w1,model.search(w1,size=10))
    # print(w2,model.search(w2,size=10))
    # w1,w2 = "harry","ron"
    # print(w1,w2,model.similarity(w1,w2))
    # names = "harry ron hermione dobby"
    # print(names,model.no_match(names.split(" ")))

    model = HierarchicalSoftmaxCBOW(C=3,ETA=.3,document="../data/人民的名义.txt")
    model.build_huffman_tree()
    model.train()
    w1,w2 = "祁同伟","高育良"
    print(w1,model.search(w1,length=3,size=10))
    print(w2,model.search(w2,length=3,size=10))
    w1,w2 = "李达康","高育良"
    print(w1,w2,model.similarity(w1,w2))
    w1,w2 = "孙连城","沙瑞金"
    print(w1,w2,model.similarity(w1,w2))
    names = "高育良 沙瑞金 李达康 刘庆祝"
    print(names,model.no_match(names.split(" ")))
