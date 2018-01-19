import numpy as np,sys
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1 + x ** 2)

def alge(x):
    return x / (np.sqrt(1+x**2))
def d_alge(x):
    return 1/ np.power((x ** 2 + 1),3/2)

def ReLu(x):
    mask = (x>0) * 1.0
    return mask * x
def d_ReLu(x):
    mask = (x>0) * 1.0
    return mask

def IDEN(x):
    return x
def d_IDEN(x):
    return 1

np.random.seed(456789)


X,Y = make_regression(n_samples=400, n_features=1, 
                                    n_informative=10, 
                                    noise=15,
                                    coef=False,tail_strength=0.4)

Y = (Y - np.min(Y)) /  (np.max(Y) - np.min(Y)) 
min_range = np.min(X)
max_range = np.max(X)

# 0. Hyper Parameter
num_epoch = 500
learning_rate = 0.0002
alpha = 0.0005
X_with_bias = np.insert(X,0,1,axis=1)
X_with_bias[:,[0, 1]] = X_with_bias[:,[1, 0]]
Y_with_dim = np.expand_dims(Y,axis=1)

# 1. Line Space to Draw Line 
theta = np.linspace(min_range , max_range , 400)
theta_with_bias = np.insert(np.expand_dims(theta,axis=1),0,1,axis=1)
theta_with_bias[:,[0, 1]] = theta_with_bias[:,[1, 0]]

# 1.5 Default Weights
w1 = np.random.randn(2,8)
w2 = np.random.randn(8,9)
w3 = np.random.randn(9,13)
w4 = np.random.randn(13,1)

# 2. Make Specific Weights
w1_l1,w2_l1,w3_l1,w4_l1 = w1,w2,w3,w4
w1_l1_reg,w2_l1_reg,w3_l1_reg,w4_l1_reg = w1,w2,w3,w4
w1_l2,w2_l2,w3_l2,w4_l2 = w1,w2,w3,w4
w1_l2_reg,w2_l2_reg,w3_l2_reg,w4_l2_reg = w1,w2,w3,w4

print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l1)
    layer_1_act = arctan(layer_1)

    layer_2 = layer_1_act.dot(w2_l1)
    layer_2_act = tanh(layer_2)

    layer_3 = layer_2_act.dot(w3_l1)
    layer_3_act = alge(layer_3)

    layer_4 = layer_3_act.dot(w4_l1)
    layer_4_act = IDEN(layer_4)

    cost = np.abs(layer_4_act-Y_with_dim).sum()  / len(X)
    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = (layer_4_act-Y_with_dim)/ len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l1.T)
    grad_3_part_2 = d_alge(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l1.T)
    grad_2_part_2 = d_tanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l1.T)
    grad_1_part_2 = d_arctan(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

    w4_l1 = w4_l1 - learning_rate * grad_4
    w3_l1 = w3_l1 - learning_rate * grad_3
    w2_l1 = w2_l1 - learning_rate * grad_2
    w1_l1 = w1_l1 - learning_rate * grad_1

# 
layer_1 = theta_with_bias.dot(w1_l1)
layer_1_act = arctan(layer_1)
layer_2 = layer_1_act.dot(w2_l1)
layer_2_act = tanh(layer_2)
layer_3 = layer_2_act.dot(w3_l1)
layer_3_act = alge(layer_3)
layer_4 = layer_3_act.dot(w4_l1)
layer_4_l1 = IDEN(layer_4)
# 


print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l2)
    layer_1_act = arctan(layer_1)

    layer_2 = layer_1_act.dot(w2_l2)
    layer_2_act = tanh(layer_2)

    layer_3 = layer_2_act.dot(w3_l2)
    layer_3_act = alge(layer_3)

    layer_4 = layer_3_act.dot(w4_l2)
    layer_4_act = IDEN(layer_4)

    cost = np.square(layer_4_act-Y_with_dim).sum() / len(X)
    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = 2*(layer_4_act-Y_with_dim) / len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l2.T)
    grad_3_part_2 = d_alge(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l2.T)
    grad_2_part_2 = d_tanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l2.T)
    grad_1_part_2 = d_arctan(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

    w4_l2 = w4_l2 - learning_rate * grad_4 
    w3_l2 = w3_l2 - learning_rate * grad_3 
    w2_l2 = w2_l2 - learning_rate * grad_2 
    w1_l2 = w1_l2 - learning_rate * grad_1
    
# 
layer_1 = theta_with_bias.dot(w1_l2)
layer_1_act = arctan(layer_1)
layer_2 = layer_1_act.dot(w2_l2)
layer_2_act = tanh(layer_2)
layer_3 = layer_2_act.dot(w3_l2)
layer_3_act = alge(layer_3)
layer_4 = layer_3_act.dot(w4_l2)
layer_4_l2 = IDEN(layer_4)
# 



# ------- reg -----------


print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l1_reg)
    layer_1_act = arctan(layer_1)

    layer_2 = layer_1_act.dot(w2_l1_reg)
    layer_2_act = tanh(layer_2)

    layer_3 = layer_2_act.dot(w3_l1_reg)
    layer_3_act = alge(layer_3)

    layer_4 = layer_3_act.dot(w4_l1_reg)
    layer_4_act = IDEN(layer_4)

    cost = np.abs(layer_4_act-Y_with_dim).sum()  / len(X)
    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = (layer_4_act-Y_with_dim)/ len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l1_reg.T)
    grad_3_part_2 = d_alge(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l1_reg.T)
    grad_2_part_2 = d_tanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l1_reg.T)
    grad_1_part_2 = d_arctan(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  


    w4_l1_reg = w4_l1_reg - learning_rate * grad_4 + alpha * np.abs(w4_l1_reg)
    w3_l1_reg = w3_l1_reg - learning_rate * grad_3 + alpha * np.abs(w3_l1_reg)
    w2_l1_reg = w2_l1_reg - learning_rate * grad_2 + alpha * np.abs(w2_l1_reg)
    w1_l1_reg = w1_l1_reg - learning_rate * grad_1 + alpha * np.abs(w1_l1_reg)
    
# 
layer_1 = theta_with_bias.dot(w1_l1_reg)
layer_1_act = arctan(layer_1)
layer_2 = layer_1_act.dot(w2_l1_reg)
layer_2_act = tanh(layer_2)
layer_3 = layer_2_act.dot(w3_l1_reg)
layer_3_act = alge(layer_3)
layer_4 = layer_3_act.dot(w4_l1_reg)
layer_4_l1_reg = IDEN(layer_4)
# 



print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l2_reg)
    layer_1_act = arctan(layer_1)

    layer_2 = layer_1_act.dot(w2_l2_reg)
    layer_2_act = tanh(layer_2)

    layer_3 = layer_2_act.dot(w3_l2_reg)
    layer_3_act = alge(layer_3)

    layer_4 = layer_3_act.dot(w4_l2_reg)
    layer_4_act = IDEN(layer_4)

    cost = np.square(layer_4_act-Y_with_dim).sum() / len(X)
    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = 2*(layer_4_act-Y_with_dim) / len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l2_reg.T)
    grad_3_part_2 = d_alge(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l2_reg.T)
    grad_2_part_2 = d_tanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l2_reg.T)
    grad_1_part_2 = d_arctan(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

    w4_l2_reg = w4_l2_reg - learning_rate * grad_4 + alpha * np.square(w4_l2_reg)
    w3_l2_reg = w3_l2_reg - learning_rate * grad_3 + alpha * np.square(w3_l2_reg)
    w2_l2_reg = w2_l2_reg - learning_rate * grad_2 + alpha * np.square(w2_l2_reg)
    w1_l2_reg = w1_l2_reg - learning_rate * grad_1 + alpha * np.square(w1_l2_reg)
    
# 
layer_1 = theta_with_bias.dot(w1_l2_reg)
layer_1_act = arctan(layer_1)
layer_2 = layer_1_act.dot(w2_l2_reg)
layer_2_act = tanh(layer_2)
layer_3 = layer_2_act.dot(w3_l2_reg)
layer_3_act = alge(layer_3)
layer_4 = layer_3_act.dot(w4_l2_reg)
layer_4_l2_reg = IDEN(layer_4)
# 







fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X,Y)
ax.plot(theta,layer_4_l1,c='r',linewidth=2,label='L1 Norm')
ax.plot(theta,layer_4_l2,c='g',linewidth=2,label='L2 Norm')
ax.plot(theta,layer_4_l1_reg,c='b',linewidth=2,label='L1 Norm with L1 Reg')
ax.plot(theta,layer_4_l2_reg,c='y',linewidth=2,label='L2 Norm with L2 Reg')
ax.legend()

plt.show()












# -- end code --