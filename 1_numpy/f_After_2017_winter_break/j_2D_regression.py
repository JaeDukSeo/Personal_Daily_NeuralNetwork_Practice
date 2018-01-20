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

def d_abs(x):
    mask = (x >= 0) *1.0
    mask2 = (x<0) * -1.0
    return mask + mask2

np.random.seed(456789)


# -2. Make training data
X,Y = make_regression(n_samples=500, 
                    n_features=1, 
                    n_informative=1, 
                    noise=0,
                    coef=False)

# -1. Outlier and min max range 
n_outliers = 50
out_liear_point = 50
X[out_liear_point:out_liear_point+n_outliers] = 1 + 0.08 * np.random.normal(size=(n_outliers, 1))
Y[out_liear_point:out_liear_point+n_outliers] = -85 + 0.8 * np.random.normal(size=n_outliers)
min_range = np.min(X)
max_range = np.max(X)

# 0. Hyper Parameter
num_epoch = 800
learning_rate = 0.0004
alpha = 3
X_with_bias = np.insert(X,0,1,axis=1)
X_with_bias[:,[0, 1]] = X_with_bias[:,[1, 0]]
Y_with_dim = np.expand_dims(Y,axis=1)

# 1. Line Space to Draw Line 
theta = np.linspace(min_range , max_range , 1000)
theta_with_bias = np.insert(np.expand_dims(theta,axis=1),0,1,axis=1)
theta_with_bias[:,[0, 1]] = theta_with_bias[:,[1, 0]]

# 1.5 Default Weights
w1 = np.random.randn(2,100)
w2 = np.random.randn(100,104)
w3 = np.random.randn(104,200)
w4 = np.random.randn(200,1)

# 2. Make Specific Weights
w1_l1,w2_l1,w3_l1,w4_l1 = w1,w2,w3,w4
w1_l2,w2_l2,w3_l2,w4_l2 = w1,w2,w3,w4

w1_l1_reg,w2_l1_reg,w3_l1_reg,w4_l1_reg = w1,w2,w3,w4
w1_l2_reg,w2_l2_reg,w3_l2_reg,w4_l2_reg = w1,w2,w3,w4

w1_l1_l2_reg,w2_l1_l2_reg,w3_l1_l2_reg,w4_l1_l2_reg = w1,w2,w3,w4
w1_l2_l1_reg,w2_l2_l1_reg,w3_l2_l1_reg,w4_l2_l1_reg = w1,w2,w3,w4


# ------ L1 Norm ------
print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l1)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2_l1)
    layer_2_act = IDEN(layer_2)

    layer_3 = layer_2_act.dot(w3_l1)
    layer_3_act = arctan(layer_3)

    layer_4 = layer_3_act.dot(w4_l1)
    layer_4_act = IDEN(layer_4)

    cost = np.abs(layer_4_act-Y_with_dim).sum()  / len(X)
    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = d_abs(layer_4_act-Y_with_dim)/ len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l1.T)
    grad_3_part_2 = d_arctan(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l1.T)
    grad_2_part_2 = d_IDEN(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l1.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

    w4_l1 = w4_l1 - learning_rate * grad_4
    w3_l1 = w3_l1 - learning_rate * grad_3
    w2_l1 = w2_l1 - learning_rate * grad_2
    w1_l1 = w1_l1 - learning_rate * grad_1

layer_1 = theta_with_bias.dot(w1_l1)
layer_1_act = tanh(layer_1)
layer_2 = layer_1_act.dot(w2_l1)
layer_2_act = IDEN(layer_2)
layer_3 = layer_2_act.dot(w3_l1)
layer_3_act = arctan(layer_3)
layer_4 = layer_3_act.dot(w4_l1)
layer_4_l1 = IDEN(layer_4)

# ------ L2 Norm  ------
print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l2)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2_l2)
    layer_2_act = IDEN(layer_2)

    layer_3 = layer_2_act.dot(w3_l2)
    layer_3_act = arctan(layer_3)

    layer_4 = layer_3_act.dot(w4_l2)
    layer_4_act = IDEN(layer_4)

    cost = np.square(layer_4_act-Y_with_dim).sum() / len(X)
    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = 2.0 * (layer_4_act-Y_with_dim) / len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l2.T)
    grad_3_part_2 = d_arctan(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l2.T)
    grad_2_part_2 = d_IDEN(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l2.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

    w4_l2 = w4_l2 - learning_rate * grad_4 
    w3_l2 = w3_l2 - learning_rate * grad_3 
    w2_l2 = w2_l2 - learning_rate * grad_2 
    w1_l2 = w1_l2 - learning_rate * grad_1
    
layer_1 = theta_with_bias.dot(w1_l2)
layer_1_act = tanh(layer_1)
layer_2 = layer_1_act.dot(w2_l2)
layer_2_act = IDEN(layer_2)
layer_3 = layer_2_act.dot(w3_l2)
layer_3_act = arctan(layer_3)
layer_4 = layer_3_act.dot(w4_l2)
layer_4_l2 = IDEN(layer_4)

#  ------ L1 Norm + L1 Reg ------
print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l1_reg)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2_l1_reg)
    layer_2_act = IDEN(layer_2)

    layer_3 = layer_2_act.dot(w3_l1_reg)
    layer_3_act = arctan(layer_3)

    layer_4 = layer_3_act.dot(w4_l1_reg)
    layer_4_act = IDEN(layer_4)

    cost = np.abs(layer_4_act-Y_with_dim).sum()  / len(X) + alpha*(np.abs(w1_l1_reg).sum() +
                                                                    np.abs(w2_l1_reg).sum() +
                                                                    np.abs(w3_l1_reg).sum() +
                                                                    np.abs(w4_l1_reg).sum()  )
    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = d_abs(layer_4_act-Y_with_dim)/ len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l1_reg.T)
    grad_3_part_2 = d_arctan(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l1_reg.T)
    grad_2_part_2 = d_IDEN(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l1_reg.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  


    w4_l1_reg = w4_l1_reg - learning_rate * (grad_4 + alpha *  d_abs(w4_l1_reg))
    w3_l1_reg = w3_l1_reg - learning_rate * (grad_3 + alpha *  d_abs(w3_l1_reg))
    w2_l1_reg = w2_l1_reg - learning_rate * (grad_2 + alpha *  d_abs(w2_l1_reg))
    w1_l1_reg = w1_l1_reg - learning_rate * (grad_1 + alpha *  d_abs(w1_l1_reg))

layer_1 = theta_with_bias.dot(w1_l1_reg)
layer_1_act = tanh(layer_1)
layer_2 = layer_1_act.dot(w2_l1_reg)
layer_2_act = IDEN(layer_2)
layer_3 = layer_2_act.dot(w3_l1_reg)
layer_3_act = arctan(layer_3)
layer_4 = layer_3_act.dot(w4_l1_reg)
layer_4_l1_reg = IDEN(layer_4)

#  ------ L2 Norm + L2 Reg ------
print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l2_reg)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2_l2_reg)
    layer_2_act = IDEN(layer_2)

    layer_3 = layer_2_act.dot(w3_l2_reg)
    layer_3_act = arctan(layer_3)

    layer_4 = layer_3_act.dot(w4_l2_reg)
    layer_4_act = IDEN(layer_4)

    cost = (np.square(layer_4_act-Y_with_dim).sum() / len(X)) + alpha * ( np.sum(w4_l2_reg ** 2)  + 
                                                                        np.sum(w3_l2_reg ** 2) +
                                                                        np.sum(w2_l2_reg ** 2) +
                                                                        np.sum(w1_l2_reg ** 2))

    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = 2*(layer_4_act-Y_with_dim) / len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l2_reg.T)
    grad_3_part_2 = d_arctan(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l2_reg.T)
    grad_2_part_2 = d_IDEN(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l2_reg.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

    w4_l2_reg = w4_l2_reg - learning_rate * (grad_4 + 2*alpha * w4_l2_reg)
    w3_l2_reg = w3_l2_reg - learning_rate * (grad_3 + 2*alpha * w3_l2_reg)
    w2_l2_reg = w2_l2_reg - learning_rate * (grad_2 + 2*alpha * w2_l2_reg)
    w1_l2_reg = w1_l2_reg - learning_rate * (grad_1 + 2*alpha * w1_l2_reg)
    
layer_1 = theta_with_bias.dot(w1_l2_reg)
layer_1_act = tanh(layer_1)
layer_2 = layer_1_act.dot(w2_l2_reg)
layer_2_act = IDEN(layer_2)
layer_3 = layer_2_act.dot(w3_l2_reg)
layer_3_act = arctan(layer_3)
layer_4 = layer_3_act.dot(w4_l2_reg)
layer_4_l2_reg = IDEN(layer_4)    

#  ------ L1 Norm + L2 Reg ------
print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l1_l2_reg)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2_l1_l2_reg)
    layer_2_act = IDEN(layer_2)

    layer_3 = layer_2_act.dot(w3_l1_l2_reg)
    layer_3_act = arctan(layer_3)

    layer_4 = layer_3_act.dot(w4_l1_l2_reg)
    layer_4_act = IDEN(layer_4)

    cost = np.abs(layer_4_act-Y_with_dim).sum()  / len(X) + alpha * ( np.sum(w4_l1_l2_reg ** 2)  + 
                                                                        np.sum(w3_l1_l2_reg ** 2) +
                                                                        np.sum(w2_l1_l2_reg ** 2) +
                                                                        np.sum(w1_l1_l2_reg ** 2))
    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = d_abs(layer_4_act-Y_with_dim)/ len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l1_l2_reg.T)
    grad_3_part_2 = d_arctan(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l1_l2_reg.T)
    grad_2_part_2 = d_IDEN(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l1_l2_reg.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  


    w4_l1_l2_reg = w4_l1_l2_reg - learning_rate * (grad_4 + 2*alpha * w4_l1_l2_reg)
    w3_l1_l2_reg = w3_l1_l2_reg - learning_rate * (grad_3 + 2*alpha * w3_l1_l2_reg)
    w2_l1_l2_reg = w2_l1_l2_reg - learning_rate * (grad_2 + 2*alpha * w2_l1_l2_reg)
    w1_l1_l2_reg = w1_l1_l2_reg - learning_rate * (grad_1 + 2*alpha * w1_l1_l2_reg)

layer_1 = theta_with_bias.dot(w1_l1_l2_reg)
layer_1_act = tanh(layer_1)
layer_2 = layer_1_act.dot(w2_l1_l2_reg)
layer_2_act = IDEN(layer_2)
layer_3 = layer_2_act.dot(w3_l1_l2_reg)
layer_3_act = arctan(layer_3)
layer_4 = layer_3_act.dot(w4_l1_l2_reg)
layer_4_l1_l2_reg = IDEN(layer_4)

#  ------ L2 Norm + L1 Reg ------
print('\n---------------------------')
for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1_l2_l1_reg)
    layer_1_act = tanh(layer_1)
    layer_2 = layer_1_act.dot(w2_l2_l1_reg)
    layer_2_act = IDEN(layer_2)
    layer_3 = layer_2_act.dot(w3_l2_l1_reg)
    layer_3_act = arctan(layer_3)
    layer_4 = layer_3_act.dot(w4_l2_l1_reg)
    layer_4_act = IDEN(layer_4)

    cost = (np.square(layer_4_act-Y_with_dim).sum() / len(X)) + alpha*(np.abs(w1_l2_l1_reg).sum() +
                                                                    np.abs(w2_l2_l1_reg).sum() +
                                                                    np.abs(w3_l2_l1_reg).sum() +
                                                                    np.abs(w4_l2_l1_reg).sum()  )

    print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = 2*(layer_4_act-Y_with_dim) / len(X)
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l2_l1_reg.T)
    grad_3_part_2 = d_arctan(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3_l2_l1_reg.T)
    grad_2_part_2 = d_IDEN(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2_l2_l1_reg.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

    w4_l2_l1_reg = w4_l2_l1_reg - learning_rate * (grad_4  + alpha *  d_abs(w4_l2_l1_reg))
    w3_l2_l1_reg = w3_l2_l1_reg - learning_rate * (grad_3  + alpha *  d_abs(w3_l2_l1_reg))
    w2_l2_l1_reg = w2_l2_l1_reg - learning_rate * (grad_2  + alpha *  d_abs(w2_l2_l1_reg))
    w1_l2_l1_reg = w1_l2_l1_reg - learning_rate * (grad_1  + alpha *  d_abs(w1_l2_l1_reg))
    
layer_1 = theta_with_bias.dot(w1_l2_l1_reg)
layer_1_act = tanh(layer_1)
layer_2 = layer_1_act.dot(w2_l2_l1_reg)
layer_2_act = IDEN(layer_2)
layer_3 = layer_2_act.dot(w3_l2_l1_reg)
layer_3_act = arctan(layer_3)
layer_4 = layer_3_act.dot(w4_l2_l1_reg)
layer_4_l2_l1_reg = IDEN(layer_4)    


print("\n")
print(w1_l1.abs().sum(),w2_l1.sum(), w3_l1.sum(),w4_l1.sum() )
print(w1_l2.sum(),w2_l2.sum(), w3_l2.sum(),w4_l2.sum())
print(w1_l1_reg.sum(),w2_l1_reg.sum(),w3_l1_reg.sum(),w4_l1_reg.sum())
print(w1_l2_reg.sum(),w2_l2_reg.sum(),w3_l2_reg.sum(),w4_l2_reg.sum())
print(w1_l1_l2_reg.sum(),w2_l1_l2_reg.sum(),w3_l1_l2_reg.sum(),w4_l1_l2_reg.sum())
print(w1_l2_l1_reg.sum(),w2_l2_l1_reg.sum(),w3_l2_l1_reg.sum(),w4_l2_l1_reg.sum())


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X,Y)
ax.plot(theta_with_bias[:,0],layer_4_l1,c='r',linewidth=1,label='L1 Norm')
ax.plot(theta_with_bias[:,0],layer_4_l2,c='g',linewidth=1,label='L2 Norm')
ax.plot(theta_with_bias[:,0],layer_4_l1_reg,c='b',linewidth=1,label='L1 Norm with L1 R eg')
ax.plot(theta_with_bias[:,0],layer_4_l2_reg,c='y',linewidth=1,label='L2 Norm with L2 Reg')
ax.plot(theta_with_bias[:,0],layer_4_l1_l2_reg,c='k',linewidth=1,label='L1 Norm with L2 Reg')
ax.plot(theta_with_bias[:,0],layer_4_l2_l1_reg,c='c',linewidth=1,label='L2 Norm with L1 Reg')
ax.legend()
plt.show()




# -- end code --





























# -- end code --