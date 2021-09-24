import numpy as np
import matplotlib.pyplot as plt

PrOriginalEp = np.zeros((2000,2000))
PrOriginalEp[1,0] = 1
PrOriginalEp[2,range(2)] = [0.5,0.5]
for i in range(3,2000):
    scale = (i-1)/2.
    x = np.arange(-(i+1)/2.+1, (i+1)/2., step=1)/scale
    y = 3./4.*(1-x**2)
    y = y/np.sum(y)
    PrOriginalEp[i, range(i)] = y
PrEp = PrOriginalEp.copy()
print(PrEp)

pr = PrEp
def get_median(a, pr=pr):
    a = np.array(a)
    x = a[~np.isnan(a)]
    n = len(x)
    weight = np.repeat(1.0, n)
    idx = np.argsort(x)
    x = x[idx]
    if n<pr.shape[0]:
        plt.plot(pr[n,:n])
        plt.plot(pr[n*2,:n*2])
        plt.plot(pr[n*5,:n*5])
        pr = pr[n,:n]
    else:
        scale = (n-1)/2.
        xxx = np.arange(-(n+1)/2.+1, (n+1)/2., step=1)/scale
        yyy = 3./4.*(1-xxx**2)
        yyy = yyy/np.sum(yyy)
        pr = (yyy*n+1)/(n+1)
    ans = np.sum(pr*x*weight) / float(np.sum(pr * weight))
    return ans


a = [10,8,5,1,23,100,6,10,55,22,7,11,89,0]
b = sorted(a)
print(a)
print(np.median(a))
print(np.mean(a))
print(get_median(a))

plt.show()
