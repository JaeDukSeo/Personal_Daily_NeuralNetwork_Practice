import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1 + x ** 2)

def alge(x):
    return x / (np.sqrt(1+x**2))
def d_alge(x):
    return 1/ np.power((x ** 2 + 1),3/2)

np.random.seed(456789)

# X,Y = make_regression(n_samples=100, n_features=2, n_informative=10, n_targets=1, bias=0.0, 
#                     effective_rank=None, tail_strength=0.5, 
#                     noise=0.0, shuffle=True, coef=False, random_state=None)

X,Y = make_regression(n_samples=200, n_features=2,n_informative=10, noise=0,
                                      coef=False,tail_strength=0)

print(X.shape)
print(Y.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1],Y, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# plt.scatter(X[:,0],X[:,1])
# plt.show()


# -- end code --