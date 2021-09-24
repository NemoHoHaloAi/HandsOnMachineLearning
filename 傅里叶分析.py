import numpy as np
import matplotlib.pyplot as plt

'''
关于对于矩形波的拟合：可以看到频率的增加为：x 3x 5x 7x，这是因为在拟合矩形波的过程中，类似2x这种频率的正弦波是不需要的

'''

x = np.array(range(-414,414,1))/100

plt.subplot(231)
y1 = 4*np.sin(x)/np.pi
plt.plot(x,y1)

plt.subplot(232)
y1 = y1+4*np.sin(3*x)/(3*np.pi)
plt.plot(x,y1)

plt.subplot(233)
y1 = y1+4*np.sin(5*x)/(5*np.pi)
plt.plot(x,y1)

plt.subplot(234)
y1 = y1+4*np.sin(7*x)/(7*np.pi)
plt.plot(x,y1)

plt.subplot(235)
y1 = y1+4*np.sin(9*x)/(9*np.pi)
plt.plot(x,y1)

plt.subplot(236)
for i in range(11,500,2):
    y1 = y1+4*np.sin(i*x)/(i*np.pi)
plt.plot(x,y1)

plt.show()
