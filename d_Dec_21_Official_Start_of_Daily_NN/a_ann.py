import numpy as np

np.random.seed(1234)

def log(x):
    return 1 / ( 1+np.exp(-x))

def d_log(x):
    return log(x) * ( 1- log(x))

# 1. Data Set 
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

# 1.5 Declare Hyper Parameters
w1 = np.random.randn(3,5)
w2 = np.random.randn(5,1)

for iter in range(100):
    
    layer_1 = x.dot(w1)
    layer_1_act = log(layer_1)
    layer_2 = layer_1_act.dot(w2)
    layer_2_act = log(layer_2)

    grad_2_part_1 = (layer_2_act - y) / (2 * len(x))
    grad_2_part_2 = d_log(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =   grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)  

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_log(layer_1)
    grad_1_part_3 = x
    grad_1 =     grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  


    w1 -= 15.0*grad_1
    w2 -= 10.0*grad_2
    


layer_1 = x.dot(w1)
layer_1_act = log(layer_1)
layer_2 = layer_1_act.dot(w2)
layer_2_act = log(layer_2)
print("FInal prediction: ", layer_2_act)
print("FInal prediction rounded : ", np.round(layer_2_act))


# -- end code --