import numpy as np,sys
from sklearn.datasets import load_digits
from scipy.ndimage.filters import maximum_filter
import skimage.measure
from scipy.signal import convolve2d
from scipy import fftpack
from sklearn.utils import shuffle


np.random.seed(12747)

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
num_epoch = 500
learning_rate = 0.0001
total_error = 0
alpha = 0.00008

w1a = np.random.randn(3,3)
w1b = np.random.randn(3,3)

w2a = np.random.randn(3,3)
w2b = np.random.randn(3,3)
w2c = np.random.randn(3,3)
w2d = np.random.randn(3,3)

w3 = np.random.randn(16,28)
w4 = np.random.randn(28,36)
w5 = np.random.randn(36,1)


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

v1a,v1b =0,0
v2a,v2b,v2c,v2d =0,0,0,0
v3,v4,v5 =0,0,0

image_matrix,image_label = shuffle(image_matrix,image_label)

image_test_label = image_label[:10]
image_label = image_label[10:]

image_test_matrix = image_matrix[:10,:,:]
image_matrix = image_matrix[10:,:,:]



print(image_matrix[0,:,:])
print(image_label[0])

print(image_matrix[110,:,:])
print(image_label[110])

print(image_matrix[122,:,:])
print(image_label[122])

print(image_matrix[3,:,:])
print(image_label[3])

print(image_label)

