import numpy as np

np.random.seed(1234)

def sigmoid(x):
    return 1/(1+ np.exp(-1* x))

def d_sidmogid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 0. Data preprocess
x = np.array([
    [0,0,1],
    [1,0,1],
    [0,1,1],
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

b1 = np.random.randn(5)
b2 = np.random.randn(6)
b3 = np.random.randn(1)

number_of_epoch = 100

v1,v2,v3 = 0,0,0


# 1. Create Graph and Train
for iter in range(number_of_epoch):
    
    layer_1 = x.dot(w1) + b1
    layer_1_act = sigmoid(layer_1)
    
    layer_2 = layer_1_act.dot(w2) + b2
    layer_2_act = sigmoid(layer_2)
    
    layer_3 = layer_2_act.dot(w3) + b3
    layer_3_act = sigmoid(layer_3)

    loss = np.square(layer_3_act-y) 

    grad_3_part_1 = (layer_3_act-y)
    grad_3_part_2 = d_sidmogid(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 = (grad_3_part_1*grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_sidmogid(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

    grad_1_part_1 = (grad_2_part_1*grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_sidmogid(layer_1)
    grad_1_part_3 = x
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    v1 = 0.9 * v1 + 2* grad_1
    v2 = 0.9 * v2 + 0.9*grad_2
    v3 = 0.9 * v3 + 0.3*grad_3

    w3 -= v3
    w2 -= v2
    w1 -= v1
    

layer_1 = x.dot(w1) + b1
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2) + b2
layer_2_act = sigmoid(layer_2)

layer_3 = layer_2_act.dot(w3) + b3
layer_3_act = sigmoid(layer_3)

print(layer_3_act)





# ------ end code ---