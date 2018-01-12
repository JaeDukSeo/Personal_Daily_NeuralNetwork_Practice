import numpy as np

def log(x):
    return 1 / (1 + np.exp(-1*x))

def d_log(x):
    return log(x) * ( 1 - log(x))

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - tanh(x) ** 2


# 0. Define Hyper Parameter
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([
    [0],
    [0],
    [0],
    [1]
])

num_epoch = 100

w1 = np.random.randn(3,6)
w2 = np.random.randn(6,1)

for iter in range(num_epoch):
    
    layer_1 = x.dot(w1)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = log(layer_2)

    cost = np.square(layer_2_act - y).sum() * 0.5

    print("Current Iter : ",iter, "  Current error: ",cost)

    grad_2_part_1 = layer_2_act - y
    grad_2_part_2 = d_log(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =   grad_2_part_3.T.dot(grad_2_part_2 * grad_2_part_1)  

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = x
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

    w1 -= grad_1
    w2 -= grad_2
    


layer_1 = x.dot(w1)
layer_1_act = tanh(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = log(layer_2)

print(layer_2_act)
print(np.round(layer_2_act))

# -- end code --