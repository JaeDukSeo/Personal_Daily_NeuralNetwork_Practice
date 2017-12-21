import numpy as np


def elu(matrix):
    mask = (matrix<=0) * 1.0
    less_zero = matrix * mask
    safe =  (matrix>0) * 1.0
    greater_zero = matrix * safe
    final = 3.0 * (np.exp(less_zero) - 1) * less_zero
    return greater_zero + final

def d_elu(matrix):
    safe = (matrix>0) * 1.0
    mask2 = (matrix<=0) * 1.0
    temp = matrix * mask2
    final = (3.0 * np.exp(temp))*mask2
    return (matrix * safe) + final

def sigmoid(x):
    return 1 / ( 1 + np.exp(-1*x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

np.random.seed(1)
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])
y = np.array([[0,0,0,1]]).T
numer_of_epoch = 1000
input_d,h1_d,h2_d,h3_d,out_d  = 3, 400, 800,600,1
learning_rate =  1

w1 = np.random.normal(size=(input_d,h1_d)) 
w2 = np.random.normal(size=(h1_d,h2_d)) 
w3 = np.random.normal(size=(h2_d,h3_d)) 
w4 = np.random.normal(size=(h3_d,out_d)) 

# w1 = np.random.randn(input_d,h1_d)
# w2 = np.random.randn(h1_d,h2_d)
# w3 = np.random.randn(h2_d,h3_d) 
# w4 = np.random.randn(h3_d,out_d) 

layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = sigmoid(layer_3)

layer_4 = layer_3_act.dot(w4)
layer_4_act = sigmoid(layer_4)
print layer_4_act

for i in range(numer_of_epoch):
    layer_1 = x.dot(w1)
    layer_1_act = sigmoid(layer_1)
    layer_1_drop = np.random.binomial(1,1-0,size = layer_1_act.shape)
    layer_1_act = layer_1_act * layer_1_drop

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = sigmoid(layer_2)
    layer_2_drop = np.random.binomial(1,1-0,size = layer_2_act.shape)
    layer_2_act = layer_2_act * layer_2_drop

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = sigmoid(layer_3)
    layer_3_drop = np.random.binomial(1,1-0,size = layer_3_act.shape)
    layer_3_act = layer_3_act * layer_3_drop

    layer_4 = layer_3_act.dot(w4)
    layer_4_act = sigmoid(layer_4)

    cost = np.square(layer_4_act - y).sum()

    grad_4_part_1 = 2.0 * (layer_4_act - y)
    grad_4_part_2 = d_sigmoid(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 = (grad_4_part_1 * grad_4_part_2).T.dot(grad_4_part_3).T

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
    grad_3_part_2 = d_sigmoid(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 = (grad_3_part_1 * grad_3_part_2).T.dot(grad_3_part_3).T

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_sigmoid(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3).T

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = x
    grad_1 = (grad_1_part_1 * grad_1_part_2).T.dot(grad_1_part_3).T

    w1 = w1 - learning_rate*grad_1
    w2 = w2 - learning_rate*grad_2
    w3 = w3 - learning_rate*grad_3
    w4 = w4 - learning_rate*grad_4

layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = sigmoid(layer_3)

layer_4 = layer_3_act.dot(w4)
layer_4_act = sigmoid(layer_4)
print layer_4_act


# ------- END FOF THE CODE -----