import numpy as np


# Func: Logistic and its functions
def logis(x):
    return 1 / ( 1+ np.exp(-1 * x))

def d_logis(x):
    return logis(x) * (1 - logis(x))

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


w1 = np.random.randn(3,5)
w2 = np.random.randn(5,1)
number_of_epoch = 100
v1 = 0
v2 = 0 
alpha = 0.8
learning_rate = 0.01

for iter in range(number_of_epoch):
    
    layer_1 = x.dot(w1)
    layer_1_act = logis(layer_1)    

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = logis(layer_2)    

    cost = np.square(layer_2 - y).sum() / ( 2 * 3)
    print("current Iter: ",iter, " cost: ", cost)

    grad_2_part_1 = layer_2 - y
    grad_2_part_2 = d_logis(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_logis(layer_1)
    grad_1_part_3 = x
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    v2 = v2 + alpha * grad_2
    v1 = v1 + alpha * grad_1

    w2 = w2 - learning_rate* v2 
    w1 = w1 - learning_rate* v1 
    



layer_1 = x.dot(w1)
layer_1_act = logis(layer_1)    

layer_2 = layer_1_act.dot(w2)
layer_2_act = logis(layer_2)    
print(layer_2_act)

    


# -- end code --