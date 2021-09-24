import numpy as np
import random,string

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


class NegativeSamplingCBOW(object):
    '''
    基于负采样的CBOW；
    1. 输入层：词向量求和后求均值；
    2. 基于负采样组成的训练set更新参数；
    3. 预测x；
    '''

    def __init__(self,NEG=5,ETA=.3,C=3,SIZE=100,M=10**8,document="../data/人民的名义.txt"):
        self.NEG = NEG
        self.ETA = ETA
        self.M = M
        self.C = C
        self.SIZE = SIZE
        self.lines = open(document).readlines()
        self.words = sum([[w for w,f in pseg.cut(line) if f not in ["m","w","d","p","vd","xc"] and w not in punc] for line in self.lines],[])
        # self.words = sum([[w for w,f in pseg.cut(line) if w not in punc] for line in self.lines],[])
        self.unique_words = list(set(self.words))
        tooless_words = [w for w in self.unique_words if self.words.count(w)<=10] # 过滤出行次数过少的词
        self.words = [w for w in self.words if w not in tooless_words]
        self.unique_words = list(set(self.words))
        self.size = len(self.words)
        self.dict_size = len(self.unique_words)
        self.word_freq = {w:self.words.count(w) for w in self.unique_words}
        self.word_freq_34 = {w:self.words.count(w)**(3./4) for w in self.unique_words}
        self.size_34 = sum([freq34 for w,freq34 in self.word_freq_34.items()])
        self.word_percentage = {w:freq34/self.size_34 for w,freq34 in self.word_freq_34.items()}
        self.word_set = sum([[w]*int(self.size*p) for w,p in self.word_percentage.items()],[])
        self.set_size = len(self.word_set)
        self.xw = np.random.random((self.dict_size,self.SIZE)) # 词向量
        self.theta = np.random.random((self.dict_size,self.SIZE)) # 词θ

    def sample(self):
        self.samples = []
        for i in range(self.C,len(self.words)-self.C-1):
            self.samples.append(self.words[i-self.C:i+self.C+1])
        self.sample_idxs = []
        for sample in self.samples:
            self.sample_idxs.append([self.unique_words.index(word) for word in sample])

    def negative_sampling(self,w):
        '''
        负采样方法：
            1. 基于词频数据构建词表数据，词在词表中的出行数量取决于它在原始文本中的词频；
            2. 将M（默认M=10^8）个小段映射到词表上，由于M>>词表大小，因此词表中的词的所占长度都会被M小段划分开；
            3. 随机从M中取NEG个数字，再取出数字对应的词表中的词；
        '''
        sample_list = []
        i = 0
        while(i<self.NEG):
            k = random.choice(range(self.M))
            idx = int(k*self.set_size/self.M)
            if self.word_set[idx] != w:
                sample_list.append(self.word_set[idx])
                i += 1
        return sample_list

    def train(self):
        self.sample()
        for words,idxs in zip(self.samples,self.sample_idxs):
            in_vector = np.array([self.xw[i] for i in idxs[:self.C]+idxs[self.C+1:]])
            # positive_vector = self.xw[idxs[self.C]].reshape((1,self.dict_size))
            positive_vector = self.xw[idxs[self.C]].reshape((1,self.SIZE))
            positive_vector_unreshape = [self.xw[idxs[self.C]]]
            positive_word = [words[self.C]]
            positive_idx = [self.unique_words.index(positive_word[0])]
            e,x_w0 = 0,positive_vector # np.mean(in_vector,axis=0)
            negative_words = self.negative_sampling(self.unique_words[idxs[self.C]])
            negative_idxs = [self.unique_words.index(neg_word) for neg_word in negative_words]
            negative_vectors = [self.xw[idx] for idx in negative_idxs]
            y = [1]+[0]*self.NEG
            set_vectors = np.array(list(positive_vector_unreshape)+list(negative_vectors))
            for yi,neg_word,neg_idx,neg_vector in zip(y,positive_word+negative_words,positive_idx+negative_idxs,set_vectors):
                # f = 1/(1+np.exp(-np.dot(x_w0.reshape((1,self.dict_size)),self.theta[neg_idx].reshape((self.dict_size,1)))[0][0]))
                f = 1/(1+np.exp(-np.dot(x_w0.reshape((1,self.SIZE)),self.theta[neg_idx].reshape((self.SIZE,1)))[0][0]))
                g = self.ETA*(yi-f)
                e = e+g*self.theta[neg_idx]
                self.theta[neg_idx] = self.theta[neg_idx]+g*x_w0
            for idx in idxs[:self.C]+idxs[self.C+1:]:
                self.xw[idx] = self.xw[idx]+e

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


if __name__ == "__main__":
    model = NegativeSamplingCBOW()
    model.train()

    w1,w2 = "李达康","高育良"
    print(w1,model.search(w1,length=3))
    print(w1,w2,model.similarity(w1,w2))
    w1,w2 = "李达康","沙瑞金"
    print(w1,w2,model.similarity(w1,w2))
    w1,w2 = "刘庆祝","沙瑞金"
    print(w1,w2,model.similarity(w1,w2))
