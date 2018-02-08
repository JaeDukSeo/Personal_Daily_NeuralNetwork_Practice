import numpy as np,sys
import tensorflow as tf
from sklearn.utils import shuffle
from scipy.signal import convolve2d
import skimage.measure
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(5678)
np.set_printoptions(precision=2,suppress=True)

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2
def ReLu(x):
    mask = (x>0) * 1.0
    return mask *x
def d_ReLu(x):
    mask = (x>0) * 1.0
    return mask 
def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))
def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1 + x ** 2)

def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()

# 1. Read Data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True).test
images,label = shuffle(mnist.images,mnist.labels)
test_image_num,training_image_num = 50,600
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]

# 2. Hyper Parameters
w1a = np.random.randn(784,512)
w1b = np.random.randn(512,256)
w1h = np.random.randn(784,256)

w2a = np.random.randn(256,128)
w2b = np.random.randn(128,64)
w2h = np.random.randn(256,64)

w3a = np.random.randn(64,32)
w3b = np.random.randn(32,10)
w3h = np.random.randn(64,10)


num_epoch = 100

for iter in range(num_epoch):
    
    for current_image_index in range(len(training_images)):
        
        current_image = np.expand_dims(training_images[current_image_index,:],axis=0)
        current_label = np.expand_dims(training_lables[current_image_index,:],axis=0)

        # 
        Rl1a  = current_image.dot(w1a)
        Rl1aA = arctan(Rl1a)
        Rl1b  = Rl1aA.dot(w1b)

        RH1  = current_image.dot(w1h)
        RH1A = tanh(RH1)

        H1 = Rl1b + RH1A
        H1A = ReLu(H1)
        # 

        # 
        R21a  = H1A.dot(w2a)
        R21aA = arctan(R21a)
        R21b  = R21aA.dot(w2b)

        RH2  = H1A.dot(w2h)
        RH2A = tanh(RH2)

        H2 = R21b + RH2A
        H2A = ReLu(H2)
        # 

        # 
        R31a  = H2A.dot(w3a)
        R31aA = arctan(R31a)
        R31b  = R31aA.dot(w3b)

        RH3  = H2A.dot(w3h)
        RH3A = tanh(RH3)

        H3 = R31b + RH3A
        H3A = ReLu(H3)
        # 

        # ---- Cost -----
        H3Soft = softmax(H3A)
        cost = ( -(current_label * np.log(H3Soft) + ( 1-current_label ) * np.log(1 - H3Soft)    )).sum() 
        # ---- Cost -----

        grad_3_part_common_part_1 = H3Soft - current_label
        grad_3_part_common_part_2 = d_ReLu(H3)

        grad_3H_part_2 = d_tanh(RH3)
        grad_3H_part_3 = H2A
        grad_3H = grad_3H_part_3.T.dot(grad_3_part_common_part_1 * grad_3_part_common_part_1 * grad_3H_part_2)

        grad_3w2_part_3 =  R31aA
        grad_3w2 =  grad_3w2_part_3.T.dot(grad_3_part_common_part_1 * grad_3_part_common_part_2) 

        print(grad_3w2.shape)
        print(w3b.shape)
        

        sys.exit()



# -- end code --