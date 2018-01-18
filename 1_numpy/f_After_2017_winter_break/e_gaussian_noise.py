import numpy as np,time
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


np.random.seed(3456789)

def tanh(x):
  return np.tanh(x)
  
def d_tanh(x):
  return 1-np.tanh(x) ** 2
  
def log(x):
  return 1/ (1 + np.exp(-1*x))
  
def d_log(x):
  return log(x) * ( 1- log(x))
  
# Func: ReLu Activation Layer just for fun  
def ReLu(x):
  mask = (x>1) * 1.0
  return x * mask

def d_ReLu(x):
  mask= (x>1) * 1.0
  return  mask
  

X, Y = make_classification(n_samples=500,n_features=2,
                        class_sep=0.04, n_redundant=0, 
                        n_informative=2,n_clusters_per_class=1)
Y = np.expand_dims(Y,axis=1)
plt.scatter(X[:,0],X[:,1],c = Y)
plt.show()

time.sleep(3)

# 0.5. Declare Hyper parameter
learing_rate = 0.03
num_epoch = 100
n_value_array = [0.01,0.3,1.0]


for n in n_value_array:
    n_value = n

    print("Current N Value : ", n_value)

    w1 = np.random.randn(2,16)
    w2 = np.random.randn(16,28)
    w3 = np.random.randn(28,35)
    w4 = np.random.randn(35,1)

    # --------- NORMAL SGD Hyper Parameter Exact Same Weight-------------
    normal_sgd_error = 100000

    w1_normal = w1
    w2_normal = w2
    w3_normal = w3
    w4_normal = w4

    # --------- ADDITIVE SGD Hyper Parameter Exact Same Weight-------------
    ADDITIVE_sgd_error = 100000

    w1_additive = w1
    w2_additive = w2
    w3_additive = w3
    w4_additive = w4


    # --------------- Training for normal---------------
    for iter in range(num_epoch):
        layer_1 = X.dot(w1_normal)
        layer_1_act = tanh(layer_1)
        
        layer_2 = layer_1_act.dot(w2_normal)
        layer_2_act = log(layer_2)
        
        layer_3 = layer_2_act.dot(w3_normal)
        layer_3_act = tanh(layer_3)
        
        layer_4 = layer_3_act.dot(w4_normal)
        layer_4_act = log(layer_4)
        
        normal_sgd_error = np.square(layer_4_act-Y).sum()  * 0.5
        # print("Normal SGD Current iter : ",iter," Current Error: ", normal_sgd_error,end=' ')
        
        grad_4_part_1 = layer_4_act-Y
        grad_4_part_2 =d_tanh(layer_4)
        grad_4_part_3 = layer_3_act
        grad_4  =grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2)
        
        grad_3_part_1 = (grad_4_part_1*grad_4_part_2).dot(w4_normal.T)
        grad_3_part_2 = d_log(layer_3)
        grad_3_part_3 = layer_2_act
        grad_3  =grad_3_part_3.T.dot(grad_3_part_1*grad_3_part_2)
        
        grad_2_part_1 = (grad_3_part_1*grad_3_part_2).dot(w3_normal.T)
        grad_2_part_2 =d_tanh(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2  =grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)
        
        grad_1_part_1 = (grad_2_part_1*grad_2_part_2).dot(w2_normal.T)
        grad_1_part_2 =d_log(layer_1)
        grad_1_part_3 = X
        grad_1  =grad_1_part_3.T.dot(grad_1_part_1*grad_1_part_2)
        
        w1_normal = w1_normal - learing_rate*grad_1
        w2_normal = w2_normal - learing_rate*grad_2
        w3_normal = w3_normal - learing_rate*grad_3
        w4_normal = w4_normal - learing_rate*grad_4

    print("Final Error for Normal SGD Current iter: ",iter,"  Error: ", np.round(normal_sgd_error,4))
    print("\n---------------------------------")

    # --------------- Training for Additive---------------
    for iter in range(num_epoch):
        layer_1 = X.dot(w1_additive)
        layer_1_act = tanh(layer_1)
        
        layer_2 = layer_1_act.dot(w2_additive)
        layer_2_act = log(layer_2)
        
        layer_3 = layer_2_act.dot(w3_additive)
        layer_3_act = tanh(layer_3)
        
        layer_4 = layer_3_act.dot(w4_additive)
        layer_4_act = log(layer_4)
        
        ADDITIVE_sgd_error = np.square(layer_4_act-Y).sum()  * 0.5
        # print("Additive Current iter : ",iter," Current Error: ", ADDITIVE_sgd_error,end='\r')
        
        grad_4_part_1 = layer_4_act-Y
        grad_4_part_2 =d_log(layer_4)
        grad_4_part_3 = layer_3_act
        grad_4  =grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_2)
        
        grad_3_part_1 = (grad_4_part_1*grad_4_part_2).dot(w4_additive.T)
        grad_3_part_2 = d_tanh(layer_3)
        grad_3_part_3 = layer_2_act
        grad_3  =grad_3_part_3.T.dot(grad_3_part_1*grad_3_part_2)
        
        grad_2_part_1 = (grad_3_part_1*grad_3_part_2).dot(w3_additive.T)
        grad_2_part_2 =d_log(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2  =grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)
        
        grad_1_part_1 = (grad_2_part_1*grad_2_part_2).dot(w2_additive.T)
        grad_1_part_2 =d_tanh(layer_1)
        grad_1_part_3 = X
        grad_1  =grad_1_part_3.T.dot(grad_1_part_1*grad_1_part_2)
        
        # ------ Calculate The Additive weight -------
        ADDITIVE_NOISE_STD = n_value / (np.power((1 + iter), 0.55))
        ADDITIVE_GAUSSIAN_NOISE = np.random.normal(loc=0,scale=ADDITIVE_NOISE_STD)
        # ------ Calculate The Additive weight -------
        
        w1_additive = w1_additive - learing_rate*(grad_1+ ADDITIVE_GAUSSIAN_NOISE)
        w2_additive = w2_additive - learing_rate*(grad_2+ ADDITIVE_GAUSSIAN_NOISE)
        w3_additive = w3_additive - learing_rate*(grad_3+ ADDITIVE_GAUSSIAN_NOISE)
        w4_additive = w4_additive - learing_rate*(grad_4+ ADDITIVE_GAUSSIAN_NOISE)
        
    print("Final Error for Additive SGD Current iter: ",iter,"  Error: ", np.round(ADDITIVE_sgd_error,4))
    print("\n---------------------------------\n\n")
















# -- end code --