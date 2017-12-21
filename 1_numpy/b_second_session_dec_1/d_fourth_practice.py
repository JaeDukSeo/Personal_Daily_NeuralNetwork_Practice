import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.model_selection import train_test_split

# 1. Create the Data 
# 2. Create the hyper parameter and other weights and bias
# 3. Train the model with the learing rate and ohter stuffs

np.random.seed(14)

n_samples = 1500
moon_x,moon_y= datasets.make_moons(n_samples=n_samples, noise=.05)
x_data, y_data, x_label, y_label = train_test_split(moon_x, moon_y, test_size=0.33)

# Func: Print out the plt
# plt.scatter(moon_x[:,0],moon_x[:,1],c=moon_y)
# plt.show()

# 1. Create the model with my own 

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - tanh(x)**2

def logistic(x):
    return 1 / ( 1 - np.exp(-1 * x))

def d_logistic(x):
    return logistic(x) * (1 -logistic(x) )


w1 = np.random.randn(2,100)
w2 = np.random.randn(100,1)

b1 = np.random.randn(100)
b2 = np.random.randn(1)

number_of_epoch = 50


plt.figure()

# 2. Train the model
for iter in range(number_of_epoch):
    
    current_x,current_label = sklearn.utils.shuffle( x_data, x_label )

    for i in range(0,len(current_x),5):

        current_x_batch = current_x[i:i+5]
        current_label_batch =   np.expand_dims(current_label[i:i+5],axis=1)

        layer_1  = current_x_batch.dot(w1) + b1
        layer_1_act = logistic(layer_1)

        layer_2  = layer_1_act.dot(w2) + b2
        layer_2_act = logistic(layer_2)

        cost = np.square(current_label_batch -layer_2_act ) / len(current_x_batch)

        grad_2_part_1 = current_label_batch -layer_2_act
        grad_2_part_2 = d_logistic(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)        

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_logistic(layer_1)
        grad_1_part_3 = current_x_batch
        grad_1 =   grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)     

        w2 = w2 - 0.01*grad_2
        w1 = w1 - 0.01*grad_1

    # Func: Display the results
    layer_1  = y_data.dot(w1) + b1
    layer_1_act = tanh(layer_1)

    layer_2  = layer_1_act.dot(w2) + b2
    layer_2_act = logistic(layer_2)   
    layer_2_act = np.squeeze(layer_2_act)

    training_loss = np.square(layer_2_act - y_label)
    print("Current epoch: ",iter," Current Error:  ", training_loss.sum())

    plt.scatter(y_data[:,0],y_data[:,1],c = layer_2_act )
    plt.pause(0.04)
        

# ---- end code ---