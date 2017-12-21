import numpy as np


def sigmoid(x):
    return 1 / ( 1+ np.exp(-(x) ) )

def d_sigmoid(x):
    return sigmoid(x) * (1 + sigmoid(x))


def LReLu(matrix):
    safe = (matrix>0) * 1.0
    mask = (matrix<=0) * 0.01
    return (safe * matrix) + (mask * matrix)

def d_LReLu(matrix):
    safe = (matrix>0) * 1.0
    mask = (matrix<=0) * 0.01
    return safe + mask    

np.random.seed(1)
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
    [1]
])
number_of_epoch = 100
input_d,h1_d,h2_d,h3_d,out_d = 3,5,6,10,1
learning_rate = 0.4

# w1 = np.random.normal(0,0.01,size=(input_d,h1_d))
# w2 = np.random.normal(-1,0.01,size=(h1_d,h2_d))
# w3 = np.random.normal(0,0.01,size=(h2_d,h3_d))
# w4 = np.random.normal(1,0.01,size=(h3_d,out_d))

w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,h3_d)
w4 = np.random.randn(h3_d,out_d)

layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = sigmoid(layer_3)

final = layer_3_act.dot(w4)
final_act = sigmoid(final)
print final_act

for i in range(number_of_epoch):

    layer_1 = x.dot(w1)
    layer_1_act = sigmoid(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = sigmoid(layer_2)

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = sigmoid(layer_3)

    final = layer_3_act.dot(w4)
    final_act = sigmoid(final)

    cost = np.square(final_act - y).sum()

    grad_4_part_1 = (final_act - y)
    grad_4_part_2 = d_sigmoid(final)
    grad_4_part_3 = layer_3_act
    # grad_4 = grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)
    grad_4 = (grad_4_part_1 * grad_4_part_2).T.dot(grad_4_part_3).T
    
    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T).T
    grad_3_part_2 = d_sigmoid(layer_3)
    grad_3_part_3 = layer_2_act
    # grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)
    grad_3 = (grad_3_part_1 * grad_3_part_2.T).dot(grad_3_part_3).T

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2.T).T.dot(w3.T).T
    grad_2_part_2 = d_sigmoid(layer_2)
    grad_2_part_3 = layer_1_act
    # grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
    grad_2 = (grad_2_part_1 * grad_2_part_2.T).dot(grad_2_part_3).T
    

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2.T).T.dot(w2.T).T
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = x
    # grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)
    grad_1 = (grad_1_part_1 * grad_1_part_2.T).dot(grad_1_part_3).T

    w1 -= learning_rate * grad_1
    w2 -= learning_rate * grad_2
    w3 -= learning_rate * grad_3
    w4 -= learning_rate * grad_4

layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = sigmoid(layer_3)

final = layer_3_act.dot(w4)
final_act = sigmoid(final)
print final_act

# ----------- END OF THE CODE ---------