import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.utils import shuffle

np.random.randn(5678)

def log_sig(x):
    return 1 / (1 + np.exp(-1*x))
def d_log_sig(x):
    return log_sig(x) * (1 - log_sig(x))

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / ( 1 + x ** 2)


digit = load_digits()

image = digit.images
label = digit.target

# 1. Prepare only one and only zero
only_zero_index = np.asarray(np.where(label == 0))
only_one_index  = np.asarray(np.where(label == 1))

# 1.5 prepare Label
only_zero_label = label[only_zero_index].T
only_one_label  = label[only_one_index].T
image_label = np.vstack((only_zero_label,only_one_label))

# 2. prepare matrix image
only_zero_image = np.squeeze(image[only_zero_index])
only_one_image = np.squeeze(image[only_one_index])
image_matrix = np.vstack((only_zero_image,only_one_image))

# 2.5 Prepare vector image and shuffle
only_zero_image_reshaped = np.reshape(only_zero_image,(-1,64))
only_one_image_reshaped = np.reshape(only_one_image,(-1,64))
image_vector = np.vstack((only_zero_image_reshaped,only_one_image_reshaped))

image_vector,image_matrix,image_label = shuffle(image_vector,image_matrix,image_label)

# 3. Split the test image and training image
test_image_num = 15

image_label_test = image_label[:test_image_num]
image_label_train = image_label[test_image_num:]

image_matrix_test = image_matrix[:test_image_num,:,:]
image_matrix_train = image_matrix[test_image_num:,:,:]

image_vector_test = image_vector[:test_image_num,:]
image_vector_train = image_vector[test_image_num:,:]

# 2. Hyper Parameters
learning_rate = 0.001
num_epoch = 100

# 3. 
w1 = np.random.randn(64,100)
w2 = np.random.randn(100,150)
w3 = np.random.randn(150,1)

for iter in range(num_epoch):
    
    layer_1 = image_vector_train.dot(w1)
    layer_1_act = arctan(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = tanh(layer_2)

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = log_sig(layer_3)

    cost = np.square(layer_3_act - image_label_train).sum() * 0.5
    print("Current iter : ",iter, "Current cost: ",cost,end='\r')

    grad_3_part_1 = layer_3_act - image_label_train
    grad_3_part_2 = d_log_sig(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3    = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)
    
    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_tanh(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2    = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
    
    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_arctan(layer_1)
    grad_1_part_3 = image_vector_train
    grad_1    = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    w1 = w1 - learning_rate * grad_1
    w2 = w2 - learning_rate * grad_2
    w3 = w3 - learning_rate * grad_3
    
layer_1 = image_vector_test.dot(w1)
layer_1_act = arctan(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = tanh(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = log_sig(layer_3)
print("\n")
print(np.round(layer_3_act))
print(image_label_test)

    

# -- end code --