import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris

'''
核方法、核函数、核技巧

指的是在优化公式中出现类似SVM中的形式：∑∑λiλjyiyjΦ(xi)Φ(xj)

上式中的Φ(xi)指的是对xi做某种线性、非线性转换，或者说特征映射操作，例如原始xi为2维，经过Φ处理后变为10维，这个过程涉及到大量的常量、向量运算，且规模正比于样本量，因此在数据量大且转换运算复杂时往往耗时很长

所谓核方法指的下列等式：
    K(xi,xj) = 〈Φ(xi),Φ(xj)〉 = Φ(xi)TΦ(xj)
    也即是函数接收原始的xi和xj作为输入，输出与xi和xj经过Φ处理后再做点积的结果一致，这个过程中省略了转换中涉及的大量运算，只关心点积结果的一致性，因此往往速度要快得多；
'''

class KernelMethod(object):
    def __init__(self):
        pass

    @staticmethod
    def linear_kernel(xi,xj):
        '''
        线性核函数
        '''
        return np.dot(xi,xj)

    @staticmethod
    def polynomial_kernel(xi,xj,times):
        '''
        多项式核函数

        times : 多项式次数
        '''
        return KernelMethod.linear_kernel(xi,xj)**times

    @staticmethod
    def gaussian_kernel(xi,xj,sigma):
        '''
        高斯核函数

        sigma : 高斯核的带宽，σ>0
        '''
        # return np.exp(-((np.linalg.norm(xi-xj,2)**2)/(2*(sigma**2))))
        return np.exp(-sigma*(np.linalg.norm(xi-xj,2)**2))

    @staticmethod
    def laplace_kernel(xi,xj,sigma):
        '''
        拉普拉斯核函数

        sigma : 拉普拉斯核的带宽，σ>0
        '''
        return np.exp(-np.linalg.norm(xi-xj,2)/sigma)

    @staticmethod
    def sigmoid_kernel(xi,xj,beta,theta):
        '''
        双曲正切核函数

        beta : β>0
        theta : θ<0
        '''
        return np.tanh(beta*np.dot(xi,xj)+theta)

    @staticmethod
    def linear_combind_kernel(a,k1,b,k2):
        '''
        对现有核函数的线性组合
        aK1(xi,xj)+bK2(xi,xj)

        a : K1系数
        k1 : K1结果
        b : K2系数
        k2 : K2结果
        '''
        return a*k1+b*k2

    @staticmethod
    def product_combind_kernel(k1,k2):
        '''
        对现有核函数的直积组合
        K1(xi,xj)*K2(xi,xj)

        k1 : K1结果
        k2 : K2结果
        '''
        return k1*k2

    @staticmethod
    def meta_combind_kernel(gi,k,gj):
        '''
        对现有核函数与普通函数的传递组合
        g(xi)*K(xi,xj)*g(xj)

        gi : g(xi)
        k : K(xi,xj)
        gj : g(xj)
        '''
        return gi*k*gj


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data[iris.target<2,:2]
    y = iris.target[iris.target<2]
    y[y==0] = -1
    X = X[::2]
    y = y[::2]

    # X = np.array([[1,1],[2,2],[3,3],[3,1],[2,3],[1,3],[2.5,2.6]])
    # y = np.array([1,1,1,1,-1,-1,-1])

    fig = plt.figure()
    ax = Axes3D(fig)

    X_kernel = np.zeros([X.shape[0],X.shape[0]])
    for i,xi in enumerate(X):
        for j,xj in enumerate(X):
            # X_kernel[i][j] = KernelMethod.linear_kernel(xi,xj)
            # X_kernel[i][j] = KernelMethod.polynomial_kernel(xi,xj,times=3)
            X_kernel[i][j] = KernelMethod.gaussian_kernel(xi,xj,sigma=.1)
            # X_kernel[i][j] = KernelMethod.laplace_kernel(xi,xj,sigma=2)
            # X_kernel[i][j] = KernelMethod.sigmoid_kernel(xi,xj,beta=.5,theta=-1)
            # a,k1 = .5,KernelMethod.linear_kernel(xi,xj)
            # b,k2 = .8,KernelMethod.gaussian_kernel(xi,xj,sigma=1)
            # X_kernel[i][j] = KernelMethod.linear_combind_kernel(a,k1,b,k2)
    XX,YY = np.meshgrid(np.array(range(X.shape[0])),np.array(range(X.shape[0])))
    ax.plot_surface(XX, YY, X_kernel, cmap=plt.cm.winter)
    plt.show()

