import numpy as np

np.random.seed(1234)



def sigmoid(x):
    return 1/(1+ np.exp(-1* x))

def d_sidmogid(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([
    [0],
    [1],
    [0],
    [1]
])

w1 = np.random.randn(3,4)
w2 = np.random.randn(4,5)
w3 = np.random.randn(5,1)

v3,v2,v1 = 0,0,0

number_of_epoch = 100
for iter in range(number_of_epoch):

    layer_1 = x.dot(w1)
    layer_1_act = sigmoid(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = sigmoid(layer_2)

    final = layer_2_act.dot(w3)
    final_act = sigmoid(final)

    cost = np.square(final_act - y) / (2 * len(x))
    print "Iter : ",iter, " cost ", cost

    grad_3_part_1 = (final_act - y)
    grad_3_part_2 = d_sidmogid(final)
    grad_3_part_3 = layer_2_act
    grad_3 =  grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_sidmogid(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =   grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)  

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_sidmogid(layer_1)
    grad_1_part_3 = x
    grad_1 =    grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)    

    v3 = 0.5 * v3 + 0.9*grad_3
    v2 = 0.5 * v2 + 0.9*grad_2
    v1 = 0.5 * v1 + 0.9*grad_1

    w3 -= v3
    w2 -= v2
    w1 -= v1
   

layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

final = layer_2_act.dot(w3)
final_act = sigmoid(final)
print final_act

# 