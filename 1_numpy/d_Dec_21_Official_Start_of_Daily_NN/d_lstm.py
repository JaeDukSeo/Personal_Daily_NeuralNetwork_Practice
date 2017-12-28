import numpy as np

np.random.seed(1234)

#  1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 
#  377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811

x = np.array([
    [1,1,2],
    [5,8,13],
    [34,55,89]
])

y = np.array([
    [3],
    [21],
    [144]
])

temp = x[:,0]
temp1 = x[:,1]

H_size = 100
X_size = 3

states = np.zeros((x.shape[0],x.shape[1] + 1))

wf = np.random.randn()
wi = np.random.randn()
wc = np.random.randn()
wo = np.random.randn()



# -- end code --