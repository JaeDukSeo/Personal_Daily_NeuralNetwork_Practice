import numpy as np,sys

# Func: Only for 2D convolution 
from scipy.signal import convolve2d

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
num_epoch = 100
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

# 4. Declare weights and hyper papermeters
w1_a = np.random.randn(3,3)
w1_b = np.random.randn(3,3)

w2_a = np.random.randn(3,3)
w2_b = np.random.randn(3,3)
w2_c = np.random.randn(3,3)
w2_d = np.random.randn(3,3)
 
w3 = np.random.randn(16,28)
w4 = np.random.randn(28,36)
w5 = np.random.randn(36,1)

for iter in range(num_epoch):
    
    for current_image_index in range(len(image_matrix)):
        
        current_image = image_matrix[current_image_index]
        current_image_label = image_label[current_image_index]

        current_image_zero_pad = np.pad(current_image,1,mode='constant')

        l1_a = convolve2d(current_image_zero_pad,w1_a,mode='valid')
        l1_aA = ReLU(l1_a)
        l1_aM = skimage.measure.block_reduce(l1_aA, block_size=(2,2), func=np.max)

        l1_b = convolve2d(current_image_zero_pad,w1_b,mode='valid')
        l1_bA = ReLU(l1_b)
        l1_bM = skimage.measure.block_reduce(l1_bA, block_size=(2,2), func=np.max)

        l2_zero_pad_a = np.pad(l1_aM,1,mode='constant')
        l2_zero_pad_b = np.pad(l1_bM,1,mode='constant')

        l2_a = convolve2d(l2_zero_pad_a,w2_a,mode='valid')
        l2_aA = ReLU(l2_a)
        l2_aM = skimage.measure.block_reduce(l2_aA, block_size=(2,2), func=np.max)

        l2_b = convolve2d(l2_zero_pad_a,w2_b,mode='valid')
        l2_bA = arctanh(l2_b)
        l2_bM = skimage.measure.block_reduce(l2_bA, block_size=(2,2), func=np.max)  
        
        l2_c = convolve2d(l2_zero_pad_b,w2_c,mode='valid')
        l2_cA = ReLU(l2_c)
        l2_cM = skimage.measure.block_reduce(l2_cA, block_size=(2,2), func=np.max)

        l2_d = convolve2d(l2_zero_pad_b,w2_d,mode='valid')
        l2_dA = log(l2_d)
        l2_dM = skimage.measure.block_reduce(l2_dA, block_size=(2,2), func=np.max)  

        l3_in_vec = np.expand_dims(np.hstack([ l2_aM.ravel(),l2_bM.ravel(),l2_cM.ravel(),l2_dM.ravel() ]),axis=0)

        l3 = l3_in_vec.dot(w3)
        l3A = tanh(l3)

        l4 = l3A.dot(w4)
        l4A = arctanh(l4)

        l5 = l4A.dot(w5)
        l5A = log(l5)

        cost = np.square(np.squeeze(l5A) - current_image_label).sum() * 0.5
        total_error = total_error + cost

        grad_5_part_1 = np.squeeze(l5A) - current_image_label
        grad_5_part_2 = d_log(l5)
        grad_5_part_3 = l4A
        grad_5 =    grad_5_part_3.T.dot(grad_5_part_1 * grad_5_part_2)     

        grad_4_part_1 = (grad_5_part_1 * grad_5_part_2).dot(w5.T)
        grad_4_part_2 = d_arctan(l4)
        grad_4_part_3 = l3A
        grad_4 =    grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)     

        grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
        grad_3_part_2 = d_tanh(l3)
        grad_3_part_3 = l3_in_vec
        grad_3 =    grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)     

        grad_2_part_1_a = np.reshape((grad_3_part_1 * grad_3_part_1).dot(w3.T)[:,:4],(2,2))
        grad_2_part_2_a = d_ReLU(l2_a)
        grad_2_part_3_a = l2_zero_pad_a
        mask = np.equal(l2_aA, l2_aM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
        grad_2_delta_a = grad_2_part_1_a.repeat(2, axis=0).repeat(2, axis=1)
        grad_2_delta_window_a = np.multiply(grad_2_delta_a, mask) * grad_2_part_2_a
        grad_2_a = convolve2d(grad_2_part_3_a,np.rot90(grad_2_delta_window_a,2),mode='valid')

        grad_2_part_1_b = np.reshape((grad_3_part_1 * grad_3_part_1).dot(w3.T)[:,4:8],(2,2))
        grad_2_part_2_b = d_ReLU(l2_b)
        grad_2_part_3_b = l2_zero_pad_a
        mask = np.equal(l2_bA, l2_bM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
        grad_2_delta_b = grad_2_part_1_b.repeat(2, axis=0).repeat(2, axis=1)
        grad_2_delta_window_b = np.multiply(grad_2_delta_b, mask) * grad_2_part_2_b
        grad_2_b = convolve2d(grad_2_part_3_b,np.rot90(grad_2_delta_window_b,2),mode='valid')

        grad_2_part_1_c = np.reshape((grad_3_part_1 * grad_3_part_1).dot(w3.T)[:,8:12],(2,2))
        grad_2_part_2_c = d_ReLU(l2_c)
        grad_2_part_3_c = l2_zero_pad_b
        mask = np.equal(l2_cA, l2_cM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
        grad_2_delta_c = grad_2_part_1_c.repeat(2, axis=0).repeat(2, axis=1)
        grad_2_delta_window_c = np.multiply(grad_2_delta_c, mask) * grad_2_part_2_c
        grad_2_c = convolve2d(grad_2_part_3_c,np.rot90(grad_2_delta_window_c,2),mode='valid')

        grad_2_part_1_d = np.reshape((grad_3_part_1 * grad_3_part_1).dot(w3.T)[:,12:],(2,2))
        grad_2_part_2_d = d_ReLU(l2_d)
        grad_2_part_3_d = l2_zero_pad_b
        mask = np.equal(l2_dA, l2_dM.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
        grad_2_delta_d = grad_2_part_1_a.repeat(2, axis=0).repeat(2, axis=1)
        grad_2_delta_window_d = np.multiply(grad_2_delta_d, mask) * grad_2_part_2_d
        grad_2_d = convolve2d(grad_2_part_3_d,np.rot90(grad_2_delta_window_d,2),mode='valid')

        grad_1_part_1_a = (grad_2_part_1_a * grad_2_part_1_b)


        # Update weghts
        w5 = w5 - learning_rate * grad_5
        w4 = w4 - learning_rate * grad_4
        w3 = w3 - learning_rate * grad_3

        w2_a = w2_a - learning_rate * grad_2_a
        w2_b = w2_b - learning_rate * grad_2_b
        w2_c = w2_c - learning_rate * grad_2_c
        w2_d = w2_d - learning_rate * grad_2_d
        
    print("Current Iter: ",iter, " current total error : ", total_error,end='\r')
    total_error = 0


        

#forward
# activationPrevious = np.copy(temp)
# activations = skimage.measure.block_reduce(temp, block_size=(2,2), func=np.max)

# print(activations)

# maxs = activations.repeat(2, axis=0).repeat(2, axis=1)
# mask = np.equal(activationPrevious, maxs).astype(int)
# delta = np.multiply(maxs, mask)


# --- end code --
