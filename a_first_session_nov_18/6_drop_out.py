import numpy as np

def sigmoid(x):
    return 1 / ( 1 + np.exp(-1*x ))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def leaky_ReLu(matrix):
    mask = (matrix<=0) * 0.1
    mask2 = (matrix>0) * 1.0
    final = mask + mask2
    return final * matrix

def d_leaky_ReLu(matrix):
    mask = (matrix<=0) * 0.1
    mask2 = (matrix>0) * 1.0
    return mask + mask2

np.random.seed(1)
x  = np.array([
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
number_of_epoch = 400
input_d,h1_d,h2_d,h3_d,out_d  = 3,500,700,300,1

w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,h3_d)
w4 = np.random.randn(h3_d,out_d)

layer_1 = x.dot(w1)
layer_1_act = leaky_ReLu(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = leaky_ReLu(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = leaky_ReLu(layer_3)

final = layer_3_act.dot(w4)
final_act = sigmoid(final)
print final_act

for numer in range(number_of_epoch):
    layer_1 = x.dot(w1)
    layer_1_act = leaky_ReLu(layer_1)
    layer_1_mask = np.random.binomial(1, 0.5, size=layer_1_act.shape)
    layer_1_act *= layer_1_mask

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = leaky_ReLu(layer_2)
    layer_2_mask = np.random.binomial(1, 0.5, size=layer_2_act.shape)
    layer_2_act *= layer_2_mask

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = leaky_ReLu(layer_3)
    layer_3_mask = np.random.binomial(1, 0.5, size=layer_3_act.shape)
    layer_3_act *= layer_3_mask

    final = layer_3_act.dot(w4)
    final_act = sigmoid(final)

    cost = np.square(final_act - y).sum()

    grad_4_part_1 = 2.0 * (final_act - y)
    grad_4_part_2 = d_leaky_ReLu(final)
    grad_4_part_3 = layer_3_act
    grad_4 = (grad_4_part_1 * grad_4_part_2).T.dot(grad_4_part_3).T

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
    grad_3_part_2 = d_leaky_ReLu(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 = (grad_3_part_1 * grad_3_part_2).T.dot(grad_3_part_3).T

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_leaky_ReLu(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3).T

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = x
    grad_1 = (grad_1_part_1 * grad_1_part_2).T.dot(grad_1_part_3).T

    w1 -= grad_1 * 0.5
    w2 -= grad_2* 0.5
    w3 -= grad_3* 0.5
    w4 -= grad_4* 0.5
    


layer_1 = x.dot(w1)
layer_1_act = leaky_ReLu(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = leaky_ReLu(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = leaky_ReLu(layer_3)

final = layer_3_act.dot(w4)
final_act = sigmoid(final)
print final_act


# -------- END OF THE CODE ---------
