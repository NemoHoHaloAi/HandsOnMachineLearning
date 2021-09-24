import sympy
from sympy.plotting import plot
from sympy.abc import N

import matplotlib.pyplot as plt

M = 10 # 有限假设空间
e = 0.1 # 泛化误差与经验误差之间的差不能大于该值
# plot(2*M*sympy.exp(-2*(e**2)*N),(N,1,1000))
# plot(4*(2**(2*N))*sympy.exp(-(1./8)*(e**2)*N),(N,1,10))
# plot(4*((2*N)**3)*sympy.exp(-(1./8)*(e**2)*N),(N,1,1000))

plt.subplot(311)
plt.plot(range(1,1000),[2.*M*sympy.E**(-2.*(e**2)*i) for i in range(1,1000)])
plt.subplot(312)
plt.plot(range(500,510),[4.*(2**(2*i))*sympy.E**(-(1./8)*(e**2)*i) for i in range(500,510)])
plt.subplot(313)
plt.plot(range(1,10000),[4.*((2*i)**3)*sympy.E**(-(1./8)*(e**2)*i) for i in range(1,10000)])
plt.show()
