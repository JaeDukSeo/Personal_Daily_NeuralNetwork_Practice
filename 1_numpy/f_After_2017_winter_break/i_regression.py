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
                                    n_informative=1, 
                                    noise=10,
                                    coef=False,tail_strength=0.4)


# ----- Special Case -----
# Frankie Case - why does the 
# cost decrease however the 
# line is just a straight line? 


# 0. Hyper Parameter
num_epoch = 3000
learning_rate = 0.000001
X_with_bias = np.insert(X,0,1,axis=1)
X_with_bias[:,[0, 1]] = X_with_bias[:,[1, 0]]
Y_with_dim = np.expand_dims(Y,axis=1)


w1 = np.random.randn(2,3)
w2 = np.random.randn(3,5)
w3 = np.random.randn(5,7)
w4 = np.random.randn(7,1)

for iter in range(num_epoch):
    
    layer_1 = X_with_bias.dot(w1)
    layer_1_act = arctan(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = IDEN(layer_2)

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = tanh(layer_3)

    layer_4 = layer_3_act.dot(w4)
    layer_4_act = IDEN(layer_4)

    cost = np.square(layer_4_act-Y_with_dim).sum() * 0.5
    # print("Current Iter: ",iter, " current cost: ",cost,end="\r")

    grad_4_part_1 = layer_4-Y_with_dim
    grad_4_part_2 = d_IDEN(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2) 

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
    grad_3_part_2 = d_tanh(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 =  (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_IDEN(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =     grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 =  (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_arctan(layer_1)
    grad_1_part_3 = X_with_bias
    grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

    w4 = w4 - learning_rate * grad_4
    w3 = w3 - learning_rate * grad_3
    w2 = w2 - learning_rate * grad_2
    w1 = w1 - learning_rate * grad_1
    
theta = np.linspace(-4 , 4 , 400)

theta_with_bias = np.insert(np.expand_dims(theta,axis=1),0,1,axis=1)
theta_with_bias[:,[0, 1]] = theta_with_bias[:,[1, 0]]

# 
layer_1 = theta_with_bias.dot(w1)
layer_1_act = arctan(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = IDEN(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = tanh(layer_3)

layer_4 = layer_3_act.dot(w4)
layer_4_theta = IDEN(layer_4)





fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X,Y)
ax.plot(theta,layer_4_theta,c='r',linewidth=3)
plt.show()


sys.exit()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1], Y)
ax.plot(theta[:,0],theta[:,1],np.squeeze(layer_4_theta),c='r',linewidth=5)
ax.plot(X[:plot_number,0],X[:plot_number,1],np.squeeze(layer_4_X),c='g')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()














# -- end code --