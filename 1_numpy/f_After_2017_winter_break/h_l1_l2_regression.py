import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

np.random.seed(34679976)

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - tanh(x) ** 2

def log(x):
    return 1 / ( 1 + np.exp(-1*x))
def d_log(x):
    return log(x) * ( 1 - log(x))

def arctanh(x):
    return np.arctan(x)
def d_arctanh(x):
    return 1 / (1 + x ** 2)

def alge(x):
    return x / (np.sqrt(1+x**2))
def d_alge(x):
    return 1/ np.power((x ** 2 + 1),3/2)

# 0. Generate Training Data
X, Y = make_classification(n_samples=500,n_features=2,
                        class_sep=4, n_repeated = 0,n_redundant=0, n_classes=2,
                        n_informative=2,n_clusters_per_class=2)
# plt.scatter(X[:,0],X[:,1],c = Y)
# plt.show()
Y = np.expand_dims(Y,axis=1)


# 1. Declare Hyper Parameters
num_epoch = 100
learing_rate = 0.01
alpha = 1.0

w1 = np.random.randn(2,25)
w2 = np.random.randn(25,57)
w3 = np.random.randn(57,150)
w4 = np.random.randn(150,1)

w1_l1 = w1
w2_l1 = w2
w3_l1 = w3
w4_l1 = w4

w1_l1_reg = w1
w2_l1_reg = w2
w3_l1_reg = w3
w4_l1_reg = w4

w1_l2 = w1
w2_l2 = w2
w3_l2 = w3
w4_l2 = w4

w1_l2_reg = w1
w2_l2_reg = w2
w3_l2_reg = w3
w4_l2_reg = w4


# ---------------- L1 Norm -------------------------------
for iter in range(num_epoch):
    
    layer_1 = X.dot(w1_l1)
    layer_1_act = arctanh(layer_1)

    layer_2 = layer_1_act.dot(w2_l1)
    layer_2_act = arctanh(layer_2)

    layer_3 = layer_2_act.dot(w3_l1)
    layer_3_act = alge(layer_3)

    layer_4 = layer_3_act.dot(w4_l1)
    layer_4_act = tanh(layer_4)

    cost = np.abs(layer_4_act - Y).sum()
    print("Current Iter: ",iter, " current error : ",cost,end='\r')

    grad_4_part_1 = layer_4_act - Y
    grad_4_part_2 = d_tanh(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4    = grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2)

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l1.T)
    grad_3_part_2 = d_alge(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3    = grad_3_part_3.T.dot(grad_3_part_1*grad_3_part_2)

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3_l1.T)
    grad_2_part_2 = d_arctanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2    = grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2_l1.T)
    grad_1_part_2 = d_arctanh(layer_1)
    grad_1_part_3 = X
    grad_1    = grad_1_part_3.T.dot(grad_1_part_1*grad_1_part_2)

    w1_l1 = w1_l1 - grad_1 * learing_rate
    w2_l1 = w2_l1 - grad_2 * learing_rate
    w3_l1 = w3_l1 - grad_3 * learing_rate
    w4_l1 = w4_l1 - grad_4 * learing_rate
    
layer_1 = X.dot(w1_l1)
layer_1_act = arctanh(layer_1)

layer_2 = layer_1_act.dot(w2_l1)
layer_2_act = arctanh(layer_2)

layer_3 = layer_2_act.dot(w3_l1)
layer_3_act = alge(layer_3)

layer_4 = layer_3_act.dot(w4_l1)
layer_4_act = tanh(layer_4)
cost = np.abs(layer_4_act - Y).sum()

print("\nFinal error : ",cost)
print("Weight Sum : ",w1_l1.sum(),w2_l1.sum(),w3_l1.sum(),w4_l1.sum())

# plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y))
# plt.show()

# plt.scatter(X[:,0],X[:,1],c = np.squeeze(layer_4_act))
# plt.show()
# ---------------- L1 Norm -------------------------------



# ---------------- L1 Norm Reg -------------------------------
for iter in range(num_epoch):
    
    layer_1 = X.dot(w1_l1_reg)
    layer_1_act = arctanh(layer_1)

    layer_2 = layer_1_act.dot(w2_l1_reg)
    layer_2_act = arctanh(layer_2)

    layer_3 = layer_2_act.dot(w3_l1_reg)
    layer_3_act = alge(layer_3)

    layer_4 = layer_3_act.dot(w4_l1_reg)
    layer_4_act = tanh(layer_4)

    cost = np.abs(layer_4_act - Y).sum()
    print("Current Iter: ",iter, " current error : ",cost,end='\r')

    grad_4_part_1 = layer_4_act - Y
    grad_4_part_2 = d_tanh(layer_4)
    grad_4_part_3 = layer_3_act
    grad_4    = grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2)

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4_l1_reg.T)
    grad_3_part_2 = d_alge(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3    = grad_3_part_3.T.dot(grad_3_part_1*grad_3_part_2)

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3_l1_reg.T)
    grad_2_part_2 = d_arctanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2    = grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2_l1_reg.T)
    grad_1_part_2 = d_arctanh(layer_1)
    grad_1_part_3 = X
    grad_1    = grad_1_part_3.T.dot(grad_1_part_1*grad_1_part_2)

    w1_l1_reg = w1_l1_reg * ( 1 - learing_rate*alpha/len(X))- learing_rate * grad_1    
    w2_l1_reg = w2_l1_reg * ( 1 - learing_rate*alpha/len(X))- learing_rate * grad_2    
    w3_l1_reg = w3_l1_reg * ( 1 - learing_rate*alpha/len(X))- learing_rate * grad_3    
    w4_l1_reg = w4_l1_reg * ( 1 - learing_rate*alpha/len(X))- learing_rate * grad_4    
    
layer_1 = X.dot(w1_l1_reg)
layer_1_act = arctanh(layer_1)

layer_2 = layer_1_act.dot(w2_l1_reg)
layer_2_act = arctanh(layer_2)

layer_3 = layer_2_act.dot(w3_l1_reg)
layer_3_act = alge(layer_3)

layer_4 = layer_3_act.dot(w4_l1_reg)
layer_4_act = tanh(layer_4)
cost = np.abs(layer_4_act - Y).sum()

print("\nFinal error : ",cost)
print("Weight Sum : ",w1_l1_reg.sum(),w2_l1_reg.sum(),w3_l1_reg.sum(),w4_l1_reg.sum())

# plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y))
# plt.show()

# plt.scatter(X[:,0],X[:,1],c = np.squeeze(layer_4_act))
# plt.show()
# ---------------- L1 Norm Reg -------------------------------

# --- end code ---