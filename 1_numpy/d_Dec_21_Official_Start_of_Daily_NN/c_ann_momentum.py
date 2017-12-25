import numpy as np

np.random.seed(1234)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - tanh(x) ** 2

x = np.array([
    [0,1,1],
    [1,0,1],
    [1,1,1],
    [0,0,1]
])

y = np.array([
    [1],
    [1],
    [0],
    [0]
])


w1 = np.random.randn(3,6)
w2 = np.random.randn(6,8)
w3 = np.random.randn(8,1)

v1,v2,v3 = 0,0,0
alpah  =0.00003
lr = 0.01

epoch = 1000

for iter in range(epoch):
    
    layer_1 = x.dot(w1)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = tanh(layer_2)
    
    layer_3 = layer_2_act.dot(w3)
    layer_3_act = tanh(layer_3)

    cost  = np.square(layer_3_act-y).sum()/len(x)

    if iter % 100 == 0 :
        print("Current iter : ",iter," current cost: ",cost)
        print((layer_3_act-y).shape)
        print((layer_3_act-np.squeeze(y)).shape)

    grad_3_part_1 = layer_3_act-y
    grad_3_part_2 = d_tanh(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =   grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)  

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_tanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =   grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2) 

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = x
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2) 

    v1 = alpah * v1 + lr * grad_1
    v2 = alpah * v2 + lr * grad_2
    v3 = alpah * v3 + lr * grad_3

    w1-= v1
    w2-= v2
    w3-= v3
    


layer_1 = x.dot(w1)
layer_1_act = tanh(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = tanh(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = tanh(layer_3)

print(layer_3_act)

print(np.round(layer_3_act))
# -- end code --