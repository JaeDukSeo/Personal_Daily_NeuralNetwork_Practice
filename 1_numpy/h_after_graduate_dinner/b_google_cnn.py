import numpy as np,sys
from sklearn.datasets import load_digits
from scipy.ndimage.filters import maximum_filter
import skimage.measure
from scipy.signal import convolve2d
from scipy import fftpack
from sklearn.utils import shuffle


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
num_epoch = 100000
learning_rate = 0.0001
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


wxg,wxc = np.random.randn(5,5),np.random.randn(5,5)
wrecg,wrecc = np.random.randn(3,3),np.random.randn(3,3)
w_full_1,w_full_2 = np.random.randn(16,128),np.random.randn(128,1)

h = np.random.randn(4,4,4)


for image_index in range(len(image_matrix)):
    
    current_image = image_matrix[image_index]
    current_image_label = image_matrix[image_index]

    cg1_h_IN = np.pad(h[0,:,:],1,mode='constant')
    c1 = convolve2d(cg1_h_IN,wrecc,mode='valid') + convolve2d(current_image,wxc,mode='valid') 
    c1A = ReLU(c1)
    g1 = convolve2d(cg1_h_IN,wrecg,mode='valid') + convolve2d(current_image,wxg,mode='valid') 
    g1A = tanh(g1)
    h[1,:,:] = g1A * h[0,:,:] + ( 1- g1A) * c1A
    
    cg2_h_IN = np.pad(h[1,:,:],1,mode='constant')
    c2 = convolve2d(cg2_h_IN,wrecc,mode='valid') + convolve2d(current_image,wxc,mode='valid') 
    c2A = ReLU(c2)
    g2 = convolve2d(cg2_h_IN,wrecg,mode='valid') + convolve2d(current_image,wxg,mode='valid') 
    g2A = tanh(g2)
    h[2,:,:] = g2A * h[1,:,:] + ( 1- g2A) * c2A
       
    cg3_h_IN = np.pad(h[2,:,:],1,mode='constant')
    c3 = convolve2d(cg3_h_IN,wrecc,mode='valid') + convolve2d(current_image,wxc,mode='valid') 
    c3A = ReLU(c3)
    g3 = convolve2d(cg3_h_IN,wrecg,mode='valid') + convolve2d(current_image,wxg,mode='valid') 
    g3A = tanh(g3)
    h[3,:,:] = g3A * h[2,:,:] + ( 1- g3A) * c3A

    full_layer_1_IN = np.expand_dims(h[3,:,:].ravel(),axis=0)
    full_layer_1 = full_layer_1_IN.dot(w_full_1)
    full_layer_1_A = tanh(full_layer_1)

    full_layer_2 = full_layer_1_A.dot(w_full_2)
    full_layer_2_A = log(full_layer_2)  

    cost = np.square(full_layer_2_A-current_image_label).sum() * 0.5

    grad_3_part_1 = full_layer_2_A-current_image_label
    grad_3_part_2 = d_log(full_layer_2)
    grad_3_part_3 = full_layer_1_A
    grad_3 =    grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2) 

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w_full_2.T)
    grad_2_part_2 = d_tanh(full_layer_1)
    grad_2_part_3 = full_layer_1_IN
    grad_2 =    grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2) 

    sys.exit()


# -- end code --