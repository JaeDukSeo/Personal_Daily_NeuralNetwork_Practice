import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(  (-1) * x)) 

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def elu(matrix):
    mask = (matrix<=0) * 1.0
    temp = mask * matrix
    mask_2 = 2.0 * (np.exp(temp)-1)
    safe = (matrix>0) * 1.0
    return (safe * matrix) + mask_2

def d_elu(matrix):
    safe = (matrix>0) * 1.0
    mask2 = (matrix<=0) * 1.0
    temp = matrix * mask2
    final = (2.0 * np.exp(temp))*mask2
    return (matrix * safe) + final

x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])
y = np.array([
    [0],
    [1],
    [1],
    [1]
])

np.random.seed(4)
input_d,h1_d,h2_d,out_d = 3,30,40,1
number_of_epoch = 1000

w1 = np.random.normal(size=(input_d,h1_d))
w2 = np.random.normal(size=(h1_d,h2_d))
w3 = np.random.normal(size=(h2_d,out_d))

layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

final = layer_2_act.dot(w3)
final_act = sigmoid(final)
print final_act>0.9

for i in range(number_of_epoch):
    layer_1 = x.dot(w1)
    layer_1_act = sigmoid(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = sigmoid(layer_2)

    final = layer_2_act.dot(w3)
    final_act = sigmoid(final)

    cost = np.square(final_act - y)

    grad_3_part_1 = 2.0 * (final_act - y)
    grad_3_part_2 = d_sigmoid(final)
    grad_3_part_3 = layer_2_act
    grad_3 = (grad_3_part_1 * grad_3_part_2).T.dot(grad_3_part_3).T

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_sigmoid(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 = (grad_2_part_1*grad_2_part_2).T.dot(grad_2_part_3).T

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = x
    grad_1 = (grad_1_part_1*grad_1_part_2).T.dot(grad_1_part_3).T

    w1 = w1 - 0.09 * grad_1
    w2 = w2 - 0.09 * grad_2
    w3 = w3 - 0.09 * grad_3

layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

final = layer_2_act.dot(w3)
final_act = sigmoid(final)
print final_act>0.9

# ------ END OF TEH CODE -----
