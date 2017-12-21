import numpy as np


np.random.seed(1234)

def ReLu(x):
    mask = (x>0) * 1.0
    return x * mask

def d_ReLu(x):
    mask  = (x<=0) * 0.0
    mask2 = (x>0) * 1.0
    return mask + mask2

def log(x):
    return 1 / (1 + np.exp(-1*x))

def d_log(x):
    return log(x) * ( 1 - log(x))


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

w1 = np.random.randn(3,5)
w2 = np.random.randn(5,6)
w3 = np.random.randn(6,1)

for iter in range(100):
    
    layer_1 = x.dot(w1)
    layer_1_act = ReLu(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = ReLu(layer_2)
  
    layer_3 = layer_2_act.dot(w3)
    layer_3_act = log(layer_3) 

    grad_3_part_1 = layer_3_act - y
    grad_3_part_2 = d_log(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_log(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_log(layer_1)
    grad_1_part_3 = x
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    w1 -= 0.7*  grad_1
    w2 -= 0.7*grad_2
    w3 -= 0.8*grad_3


layer_1 = x.dot(w1)
layer_1_act = ReLu(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = ReLu(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = log(layer_3) 
print(layer_3_act)

# -- end code --