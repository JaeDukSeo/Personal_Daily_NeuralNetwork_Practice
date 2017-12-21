import numpy as np


def relu(matrix):
    mask = (matrix>0)
    return matrix * mask

def d_relu(matrix):
    mask = (matrix>0)
    return 1.0 * mask


def leaky_Relu(matrix):
    mask1 =  (matrix<=0) * 0.01 
    mask2 = (mask1 == 0)
    final = mask1 + mask2
    return matrix * final

def d_leaky_Relu(matrix):
    mask1 = (matrix<=0) * 0.01
    mask2 = (matrix>0) 
    return mask1+mask2

def sigmoid(x):
    return 1 / (1 + np.exp(  (-1) * x)) 

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))



np.random.seed(1)
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
    ])
y = np.array([ [0],[0],[0],[1]] )
learning_rate = 0.3
numpber_of_epoch = 100
input_d ,h1d,out_d = 3,3,1

w1 = 2*np.random.randn(3,3)-3
w2 = 2*np.random.randn(3,1)-6

layer_1 = x.dot(w1)
layer_1_act = leaky_Relu(layer_1)
final = layer_1_act.dot(w2)
final_act = sigmoid(final)
print final_act

for turn in range(numpber_of_epoch):

    for i in range(100):
        layer_1 = x.dot(w1)
        layer_1_act = leaky_Relu(layer_1)

        final = layer_1_act.dot(w2)
        final_act = sigmoid(final)

        cost_loss = np.square(final_act - y).sum()

        grad_2_part_1 = 2.0 * (final_act - y)
        grad_2_part_2 = d_sigmoid(final)
        grad_2_part_3 = layer_1_act
        grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3).T

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_leaky_Relu(layer_1)
        grad_1_part_3 = x
        grad_1 = (grad_1_part_1 * grad_1_part_2).T.dot(grad_1_part_3).T

        w1 = w1 - learning_rate * grad_1
        w2 = w2 - learning_rate * grad_2



layer_1 = x.dot(w1)
layer_1_act = leaky_Relu(layer_1)
final = layer_1_act.dot(w2)
final_act = sigmoid(final)
print final_act















#  ----------- End of the code ---------