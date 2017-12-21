import numpy as np


def ReLu(x):
    x[x<0]=0
    return x

def d_ReLu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(  (-1) * x)) 

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


np.random.seed(3)
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
    ])
y = np.array([ [0],[0],[0],[1]] )
epoch  = 100
learning_rate = 0.5

w1 = np.random.randn(3,4)
w2 = np.random.randn(4,1)

for i in range(epoch):

    for ii in range(100):
        layer_1 = x.dot(w1)
        layer_1_act = sigmoid(layer_1)

        layer_2 = layer_1_act.dot(w2)
        final = sigmoid(layer_2)

        cost_losss = np.square(final - y).sum()

        grad_2_part_1 = 2.0 * (final - y)
        grad_2_part_2 = d_sigmoid(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3).T

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = x
        grad_1 = (grad_1_part_1 * grad_1_part_2).dot(grad_1_part_3).T

        w1 = w1 - learning_rate * grad_1
        w2 = w2 - learning_rate * grad_2
        


layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)
layer_2 = layer_1_act.dot(w2)
final = sigmoid(layer_2)
print  final
print '---------------------'

w1 = np.random.randn(3,3)
w2 = np.random.randn(3,1)

for i in range(epoch):

    for ii in range(100):
        layer_1 = x.dot(w1)
        layer_1_act = sigmoid(layer_1)

        layer_2 = layer_1_act.dot(w2)
        final = sigmoid(layer_2)

        cost_losss = np.square(final - y).sum()

        grad_2_part_1 = 2.0 * (final - y)
        grad_2_part_2 = d_sigmoid(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3).T

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = x

        grad_1 = (grad_1_part_1 * grad_1_part_2).T.dot(grad_1_part_3).T

        w1 = w1 - learning_rate * grad_1
        w2 = w2 - learning_rate * grad_2
        


layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)
layer_2 = layer_1_act.dot(w2)
final = sigmoid(layer_2)
print  final



# 2. Create weight
# w1 = np.random.randn(input_d, hidden1_d) 
# w2 = np.random.randn(hidden1_d, out_d) 

# for i in range(number_of_epoch):

#     for ii in range(10):
#         layer_1 = x.dot(w1)
#         layer_1_sig = sigmoid(layer_1)

#         layer_2 = layer_1_sig.dot(w2)
#         layer_2_sig = sigmoid(layer_2)

#         error_cost = np.square(layer_2_sig - y)

#         grad_2_part_1 = 2.0 * (layer_2_sig - y)
#         grad_2_part_2 = d_sigmoid(layer_2)
#         grad_2_part_3 = layer_1_sig
#         grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3).T

#         grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
#         grad_1_part_2 = d_sigmoid(layer_1)
#         grad_1_part_3 = x
#         grad_1 = (grad_1_part_1 * grad_1_part_2).T.dot(grad_1_part_3)

#         w1 = w1 - learning_rate * grad_1
#         w2 = w2 - learning_rate * grad_2






# --------- END OF THE CODE --------