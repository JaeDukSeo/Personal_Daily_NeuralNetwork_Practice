import numpy as np,sys

# Func: Only for 2D convolution 
from scipy.signal import convolve2d
from sklearn.utils import shuffle

# Func: For Back propagation on Max Pooling
from scipy.ndimage.filters import maximum_filter
import skimage.measure
from sklearn.datasets import load_digits

np.random.seed(12314)

def ReLU(x):
    mask  = (x >0) * 1.0 
    return mask * x
def d_ReLU(x):
    mask  = (x >0) * 1.0 
    return mask 

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def arctanh(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / ( 1 + x ** 2)

def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))

# 1. Prepare Data
data =load_digits()
image = data.images
label = data.target
num_epoch = 501
learning_rate =0.003
total_error = 0

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


image_matrix,image_label = shuffle(image_matrix,image_label)

image_test_label = image_label[:10]
image_label = image_label[10:]

image_test_matrix = image_matrix[:10,:,:]
image_matrix = image_matrix[10:,:,:]












w1 = np.random.randn(3,3) * 0.2
w2 = np.random.randn(3,3)* 0.2
w3 = np.random.randn(16,1)* 0.2

for iter in range(num_epoch):

    for current_image_index in range(len(image_matrix)):
        
        current_image = image_matrix[current_image_index]
        current_image_label = image_label[current_image_index]

        l1 = convolve2d(current_image,w1,mode='valid')
        l1A = arctanh(l1)

        l2 = convolve2d(l1A,w2,mode='valid')
        l2A = arctanh(l2)

        l3IN = np.expand_dims(l2A.ravel(),0)

        l3 = l3IN.dot(w3)
        l3A = log(l3)

        cost = np.square(l3A - current_image_label).sum() * 0.5
        total_error += cost

        grad_3_part_1 = l3A - current_image_label
        grad_3_part_2 = d_log(l3)
        grad_3_part_3 =l3IN
        grad_3 =  grad_3_part_3.T.dot( grad_3_part_1 * grad_3_part_2)

        grad_2_part_1 = np.reshape((grad_3_part_1 * grad_3_part_2).dot(w3.T),(4,4))
        grad_2_part_2 = d_arctan(l2)
        grad_2_part_3 = l1A
        grad_2=  np.rot90( convolve2d(grad_2_part_3,grad_2_part_1 * grad_2_part_2,mode='valid')     ,2)
        
        grad_1_part_IN = np.pad(w2,3,mode='constant')
        grad_1_part_IN2 = np.rot90(grad_2_part_1 * grad_2_part_2,2)

        grad_1_part_1 = convolve2d(grad_1_part_IN,grad_1_part_IN2,mode='valid')
        grad_1_part_2 = d_arctan(l1)
        grad_1_part_3 = current_image
        grad_1 =  np.rot90( convolve2d(grad_1_part_3,grad_1_part_1 * grad_1_part_2,mode='valid')     ,2)

        w1 = w1 - learning_rate * grad_1
        w2 = w2 - learning_rate * grad_2
        w3 = w3 - learning_rate * grad_3

    if iter % 100 == 0:
        print("\n")
        predictions = np.array([])
        for image_index in range(len(image_test_matrix)):

            current_image  = image_test_matrix[image_index]

            l1 = convolve2d(current_image,w1,mode='valid')
            l1A = arctanh(l1)

            l2 = convolve2d(l1A,w2,mode='valid')
            l2A = arctanh(l2)

            l3IN = np.expand_dims(l2A.ravel(),0)

            l3 = l3IN.dot(w3)
            l3A = log(l3)

            predictions = np.append(predictions,l3A)


        print('-------')
        print(image_test_label.T)
        print('-------')
        print(predictions.T)
        print('-------')
        print(np.round(predictions).T)
        print("\n")

        
    print('Current iter:  ', iter, ' current cost: ', cost, end='\r')
    total_error = 0

        
    
