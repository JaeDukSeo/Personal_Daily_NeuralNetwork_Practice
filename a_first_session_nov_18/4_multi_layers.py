import numpy as np

def LReLu(matrix):
    mask =  (matrix<=0) * 0.01
    mask2 = (matrix>0) * 1.0
    final_mask = mask + mask2
    return final_mask * matrix

def d_LReLu(matrix):
    mask =  (matrix<=0) * 0.01
    mask2 = (matrix>0) * 1.0
    return mask + mask2

def sigmoid(x):
    return 1 / (1 + np.exp( -1 * x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

np.random.seed(3)

# 0. Data preprocess and declare hyper parameteres
number_of_epoch = 1000
input_d, h1_d,h2_d,h3_d,out_d = 3,4,4,5,1
learning_rate = 0.05
x = np.array([ 
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])
y = np.array([
    [0],[0],[0],[1]
]) 

w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,h3_d)
w4 = np.random.randn(h3_d,out_d)

# for i in range(number_of_epoch):

    # for ii in range(40):
cost = 0.1
itter = 0
while cost > 0.05:

    layer_1 = x.dot(w1)
    layer_1_act = LReLu(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = LReLu(layer_2)

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = LReLu(layer_3)

    final = layer_3_act.dot(w4)
    final_act = sigmoid(final)

    cost = np.square(final_act - y).sum()
    print itter,'iteration : ', cost
    itter = itter + 1

    grad_4_part_1 = 2.0 * (final_act - y)
    grad_4_part_2 = d_sigmoid(final)
    grad_4_part_3 = layer_3_act
    grad_4 = (grad_4_part_1 * grad_4_part_2).T.dot(grad_4_part_3).T

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
    grad_3_part_2 = d_LReLu(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 = (grad_3_part_1 * grad_3_part_2).T.dot(grad_3_part_3).T

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_LReLu(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3).T

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_LReLu(layer_1)
    grad_1_part_3 = x
    grad_1 = (grad_1_part_1 * grad_1_part_2).T.dot(grad_1_part_3).T

    w1 = w1 - learning_rate * grad_1
    w2 = w2 - learning_rate * grad_2
    w3 = w3 - learning_rate * grad_3
    w4 = w4 - learning_rate * grad_4


layer_1 = x.dot(w1)
layer_1_act = LReLu(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = LReLu(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = LReLu(layer_3)

final = layer_3_act.dot(w4)
final_act = sigmoid(final)
print final_act























# --------------- END OF THE COE -------------