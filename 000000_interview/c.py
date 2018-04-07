import numpy as np
import matplotlib.pyplot as plt

def one(x): return 1/x
def two(x): return -1/(x^2)

data = np.array([2,3,4,5,6,7,8,9,10,11])
one_a = one(data)
two_a = two(data)
plt.plot(range(len(data)),one_a,'r')
plt.plot(range(len(data)),two_a,'b')
plt.show()
print('-------------')


def log(x): return np.log(x)
def log2(x): return np.log(x+10)

data = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
one_a =log(data)
two_a = log2(data)
plt.plot(range(len(data)),one_a,'r')
plt.plot(range(len(data)),two_a,'b')
plt.show()
print('-------------')












# -- end code --