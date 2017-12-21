from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt,numpy as np
from mpl_toolkits.mplot3d import Axes3D



def sigmoid(x):
    return 1 / (1 + np.exp( -1*x ))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 0. Data Preprocess and Declare hyperparameters
np.random.seed(1)

X, y = make_moons(n_samples=1500, random_state=40, noise=0.05)
x_data, y_data, x_label, y_label = train_test_split(X, y, random_state=50)
x_label = np.expand_dims(x_label,axis=1)
y_label = np.expand_dims(y_label,axis=1)
# plt.scatter(x_data[:,0], x_data[:,1],c=x_label)
# plt.show()

x_data = x_data[:400]
x_label = x_label[:400]
number_of_epoch = 600
input_d ,h1_d,h2_d,h3_d, out_d  = 2,800,500,400,1
learning_rates = [0.001,0.1,1,10,1000]

w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,h3_d)
w4 = np.random.randn(h3_d,out_d)

# 1. Declare the operations
layer_1 = x_data.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = sigmoid(layer_3)

final = layer_3_act.dot(w4)
final_act = sigmoid(final)
print (final_act - x_label).sum() / len(final_act)

for iter in range(number_of_epoch):
    layer_1 = x_data.dot(w1)
    layer_1_act = sigmoid(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = sigmoid(layer_2)

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = sigmoid(layer_3)

    final = layer_3_act.dot(w4)
    final_act = sigmoid(final)

    if iter%100 == 0:
        print "Error : ",(final_act - x_label).sum() / len(final_act)

    grad_4_part_1  = (final_act - x_label)
    grad_4_part_2  = d_sigmoid(final)
    grad_4_part_3  = layer_3_act
    grad_4 = grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_1)

    grad_3_part_1  = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
    grad_3_part_2  = d_sigmoid(layer_3)
    grad_3_part_3  = layer_2_act
    grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1  = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2  = d_sigmoid(layer_2)
    grad_2_part_3  = layer_1_act
    grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

    grad_1_part_1  = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2  = d_sigmoid(layer_1)
    grad_1_part_3  = x_data
    grad_1  = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)
    
    w4 -= 0.5 * grad_4
    w3 -= 0.5 * grad_3
    w2 -= 0.5 * grad_2
    w1 -= 0.5 * grad_1
    

layer_1 = x_data.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = sigmoid(layer_3)

final = layer_3_act.dot(w4)
final_act = sigmoid(final)
print (final_act - x_label).sum() / len(final_act)








# ------- End of the code -------