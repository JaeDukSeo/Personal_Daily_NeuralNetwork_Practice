import numpy as np


np.random.seed(1234)

def log(x):
    return 1/ (1 + np.exp(-1 *x))

def d_log(x):
    return log(x) * ( 1- log(x))

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

x = np.array([
    [0,0,1],
    [1,0,1],
    [0,1,1],
    [1,1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


w1 = np.random.randn(3,5)
w2 = np.random.randn(5,6)
w3 = np.random.randn(6,1)

v1,v2,v3 = 0,0,0
lr = 0.1
alpha = 0.01
num_epoch = 1000

for iter in range(num_epoch):
    
    layer_1 = x.dot(w1)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = log(layer_2)

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = tanh(layer_3)

    cost = np.square(layer_3_act-y).sum() * 0.5
    print("Current Iter: ",iter, " Current Cost : ", cost)

    grad_3_part_1 = layer_3_act-y
    grad_3_part_2 = d_tanh(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3   = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_log(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2   =grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = x
    grad_1  =grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    v1 = alpha * v1 + lr * grad_1
    v2 = alpha * v2 + lr * grad_2
    v3 = alpha * v3 + lr * grad_3

    w1-=v1
    w2-=v2
    w3-=v3
    

layer_1 = x.dot(w1)
layer_1_act = tanh(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = log(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = tanh(layer_3)
print(layer_3_act)

# -- end code --