import numpy as np,sys
from scipy.signal import convolve2d
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle

np.random.seed(567)

def elu(matrix):
    mask = (matrix<=0) * 1.0
    less_zero = matrix * mask
    safe =  (matrix>0) * 1.0
    greater_zero = matrix * safe
    final = 3.0 * (np.exp(less_zero) - 1) * less_zero
    return greater_zero + final

def d_elu(matrix):
    safe = (matrix>0) * 1.0
    mask2 = (matrix<=0) * 1.0
    temp = matrix * mask2
    final = (3.0 * np.exp(temp))*mask2
    return (matrix * safe) + final

mnist = input_data.read_data_sets('../MNIST_DATA', one_hot=False)
temp = mnist.test

# 0. Load Data
images, labels = temp.images, temp.labels
num_epoch = 1
learning_rate = 0.001
value  = 0.002


# 1. Declare Weights and hyper parameter
D_w0 = np.random.randn(3,3) * value
D_w1 = np.random.randn(3,3)* value

D_w2a = np.random.randn(3,3)* value
D_w2b = np.random.randn(3,3)* value

D_w3a = np.random.randn(3,3)* value
D_w3b = np.random.randn(3,3)* value

D_w4a = np.random.randn(3,3)* value
D_w4b = np.random.randn(3,3)* value
D_w4c = np.random.randn(3,3)* value

D_w5a = np.random.randn(3,3)* value
D_w5b = np.random.randn(3,3)* value
D_w5c = np.random.randn(3,3)* value

D_w6a = np.random.randn(3,3)* value
D_w6b = np.random.randn(3,3)* value
D_w6c = np.random.randn(3,3)* value



for iter in range(num_epoch):
    
    for current_image in images:

        image_reshape = np.reshape(current_image,(28,28))

        D_0Input = np.pad(image_reshape,1,mode='constant')
        D_0l = convolve2d(D_0Input,D_w0,mode='valid')
        D_l0A = elu(D_0l)

        D_1Input = np.pad(D_l0A,1,mode='constant')
        D_1l = convolve2d(D_1Input,D_w1,mode='valid')
        D_1lA = elu(D_1l)

        print(D_1lA.shape)

        sys.exit()


        D_2Input = np.pad(D_1lA,1,mode='constant')
        D_2l_a = convolve2d(D_2Input,D_w2a,mode='constant')
        D_2lA_a = elu(D_2l_a)
        D_2l_b = convolve2d(D_2Input,D_w2b,mode='constant')
        D_2lA_b = elu(D_2l_b)

        sys.exit()

# -- end code --