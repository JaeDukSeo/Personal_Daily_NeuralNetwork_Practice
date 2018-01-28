import numpy as np,sys
from sklearn.datasets import load_digits
from scipy.ndimage.filters import maximum_filter
import skimage.measure
from scipy.signal import convolve2d
from scipy import fftpack
from sklearn.utils import shuffle


np.random.seed(1234414)

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
learning_rate = 0.09
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

v1a,v1b =0,0
v2a,v2b,v2c,v2d =0,0,0,0
v3,v4,v5 =0,0,0
alpha = 0.0009

image_matrix,image_label = shuffle(image_matrix,image_label)

image_test_label = image_label[:10]
image_label = image_label[10:]

image_test_matrix = image_matrix[:10,:,:]
image_matrix = image_matrix[10:,:,:]

w1a = np.random.randn(3,3)
w1b = np.random.randn(3,3)

w2a = np.random.randn(3,3)
w2b = np.random.randn(3,3)
w2c = np.random.randn(3,3)
w2d = np.random.randn(3,3)

w3 = np.random.randn(16,20)
w4 = np.random.randn(20,36)
w5 = np.random.randn(36,1)

for iter in range(num_epoch):
    for image_index in range(len(image_matrix)):
        
        current_image = image_matrix[image_index]
        current_image_label = image_label[image_index]

        l1aIN,l1bIN = np.pad(current_image,1,mode='constant'),np.pad(current_image,1,mode='constant')

        l1a = convolve2d(l1aIN,w1a,mode='valid')
        l1aA = ReLU(l1a)
        l1aM = skimage.measure.block_reduce(l1aA, block_size=(2,2), func=np.max)

        l1b = convolve2d(l1bIN,w1b,mode='valid')
        l1bA = arctanh(l1b)
        l1bM = skimage.measure.block_reduce(l1bA, block_size=(2,2), func=np.max)

        l2aIN,l2bIN =  np.pad(l1aM,1,mode='constant'),np.pad(l1aM,1,mode='constant')
        l2cIN,l2dIN =  np.pad(l1bM,1,mode='constant'),np.pad(l1bM,1,mode='constant')

        l2a = convolve2d(l2aIN,w1b,mode='valid')
        l2aA = arctanh(l2a)
        l2aM = skimage.measure.block_reduce(l2aA, block_size=(2,2), func=np.max) 

        l2b = convolve2d(l2bIN,w1b,mode='valid')
        l2bA = ReLU(l2b)
        l2bM = skimage.measure.block_reduce(l2bA, block_size=(2,2), func=np.max)

        l2c = convolve2d(l2cIN,w1b,mode='valid')
        l2cA = arctanh(l2c)
        l2cM = skimage.measure.block_reduce(l2cA, block_size=(2,2), func=np.max)

        l2d = convolve2d(l2dIN,w1b,mode='valid')
        l2dA = tanh(l2d)
        l2dM = skimage.measure.block_reduce(l2dA, block_size=(2,2), func=np.max)

        l3IN = np.expand_dims(np.hstack([l2aM.ravel(),l2bM.ravel(),l2cM.ravel(),l2dM.ravel() ]),axis=0)

        l3 = l3IN.dot(w3)
        l3A = arctanh(l3)

        l4 = l3A.dot(w4)
        l4A = tanh(l4)

        l5 = l4A.dot(w5)
        l5A = log(l5)

        cost = np.square(l5A - current_image_label).sum() * 0.5
        total_error += cost

        grad_5_part_1 = l5A - current_image_label
        grad_5_part_2 = d_log(l5)
        grad_5_part_3 =l4A
        grad_5 =grad_5_part_3.T.dot(grad_5_part_1 * grad_5_part_2)

        grad_4_part_1 = (grad_5_part_1 * grad_5_part_2).dot(w5.T)
        grad_4_part_2 = d_tanh(l4)
        grad_4_part_3 =l3A
        grad_4 =  grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2 )

        grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
        grad_3_part_2 = d_arctan(l3)
        grad_3_part_3 =l3IN
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)  

        grad_2_part_IN = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        
        grad_2_window_a = np.reshape(grad_2_part_IN[:,:4],(2,2))
        grad_2_mask_a =  np.equal(l2aA, l2aM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
        grad_2_part_1_a =  grad_2_mask_a *  grad_2_window_a.repeat(2, axis=0).repeat(2, axis=1)
        grad_2_part_2_a = d_arctan(l2a)
        grad_2_part_3_a = l2aIN
        grad_2_a = np.rot90(convolve2d(grad_2_part_3_a,np.rot90(grad_2_part_2_a * grad_2_part_1_a,2),mode='valid'),2)

        grad_2_window_b = np.reshape(grad_2_part_IN[:,4:8],(2,2))
        grad_2_mask_b =  np.equal(l2bA, l2bM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
        grad_2_part_1_b = grad_2_mask_b *  grad_2_window_b.repeat(2, axis=0).repeat(2, axis=1)
        grad_2_part_2_b = d_ReLU(l2b)
        grad_2_part_3_b = l2bIN
        grad_2_b = np.rot90(convolve2d(grad_2_part_3_b,np.rot90(grad_2_part_2_b * grad_2_part_1_b,2),mode='valid'),2)

        grad_2_window_c = np.reshape(grad_2_part_IN[:,8:12],(2,2))
        grad_2_mask_c =  np.equal(l2cA, l2cM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
        grad_2_part_1_c = grad_2_mask_c *  grad_2_window_c.repeat(2, axis=0).repeat(2, axis=1)
        grad_2_part_2_c = d_arctan(l2c)
        grad_2_part_3_c = l2cIN
        grad_2_c = np.rot90(convolve2d(grad_2_part_3_c,np.rot90( grad_2_part_2_c * grad_2_part_1_c,2),mode='valid'),2)

        grad_2_window_d = np.reshape(grad_2_part_IN[:,12:],(2,2))
        grad_2_mask_d =  np.equal(l2dA, l2dM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
        grad_2_part_1_d =  grad_2_mask_d *  grad_2_window_d.repeat(2, axis=0).repeat(2, axis=1)
        grad_2_part_2_d = d_tanh(l2d)
        grad_2_part_3_d = l2dIN
        grad_2_d = np.rot90(convolve2d(grad_2_part_3_d,np.rot90( grad_2_part_2_d * grad_2_part_1_d,2),mode='valid'),2)
                    
        grad_1_part_IN_a =np.rot90(  grad_2_part_1_a * grad_2_part_2_a ,2)
        grad_1_part_IN_a_padded = np.pad(w2a,2,mode='constant')
        grad_1_part_a = convolve2d(grad_1_part_IN_a_padded,grad_1_part_IN_a,mode='valid')    

        grad_1_part_IN_b = np.rot90( grad_2_part_2_b * grad_2_part_1_b,2)
        grad_1_part_IN_b_padded = np.pad(w2b,2,mode='constant')
        grad_1_part_b = convolve2d(grad_1_part_IN_b_padded,grad_1_part_IN_b,mode='valid')    

        grad_1_window_a = grad_1_part_a + grad_1_part_b
        grad_1_mask_a =  np.equal(l1aA, l1aM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
        grad_1_part_1_a =  grad_1_mask_a *  grad_1_window_a.repeat(2, axis=0).repeat(2, axis=1) 
        grad_1_part_2_a = d_ReLU(l1a)
        grad_1_part_3_a = l1aIN
        grad_1_a = np.rot90(convolve2d(grad_1_part_3_a,np.rot90( grad_1_part_1_a *grad_1_part_2_a,2),mode='valid'),2)
        
        grad_1_part_IN_c = np.rot90(grad_2_part_1_c * grad_2_part_2_c,2)
        grad_1_part_IN_c_padded = np.pad(w2c,2,mode='constant')
        grad_1_part_c = convolve2d(grad_1_part_IN_c_padded,grad_1_part_IN_c,mode='valid')    

        grad_1_part_IN_d = np.rot90(grad_2_part_1_d * grad_2_part_2_d ,2)
        grad_1_part_IN_d_padded = np.pad(w2d,2,mode='constant')
        grad_1_part_d = convolve2d(grad_1_part_IN_d_padded,grad_1_part_IN_d,mode='valid')    

        grad_1_window_b = grad_1_part_c + grad_1_part_d
        grad_1_mask_b =  np.equal(l1bA, l1bM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
        grad_1_part_1_b =  grad_1_mask_b *  grad_1_window_b.repeat(2, axis=0).repeat(2, axis=1) 
        grad_1_part_2_b = d_arctan(l1b)
        grad_1_part_3_b = l1bIN
        grad_1_b = np.rot90(convolve2d(grad_1_part_3_b, np.rot90(grad_1_part_1_b *grad_1_part_2_b,2),mode='valid'),2)
        
        v5 = alpha * v5 + learning_rate * grad_5 
        v4 = alpha * v4 + learning_rate * grad_4 
        v3 = alpha * v3 + learning_rate * grad_3 

        v2a = alpha * v2a + learning_rate * grad_2_a 
        v2b = alpha * v2b + learning_rate * grad_2_b 
        v2c = alpha * v2c + learning_rate * grad_2_c 
        v2d = alpha * v2d + learning_rate * grad_2_d 

        v1b = alpha * v1b + learning_rate * grad_1_b  
        v1a = alpha * v1a + learning_rate * grad_1_a
        
        w5 = w5 - v5  
        w4 = w4 - v4 
        w3 = w3 - v3
            
        w2d = w2d - v2d  
        w2c = w2c - v2c
        w2b = w2b - v2b
        w2a = w2a - v2a  

        w1b = w1b - v1b 
        w1a = w1a - v1a  

        # w5 = w5 - learning_rate * grad_5    
        # w4 = w4 - learning_rate * grad_4    
        # w3 = w3 - learning_rate * grad_3
            
        # w2d = w2d - learning_rate * grad_2_d    
        # w2c = w2c - learning_rate * grad_2_c  
        # w2b = w2b - learning_rate * grad_2_b    
        # w2a = w2a - learning_rate * grad_2_a    

        # w1b = w1b - learning_rate * grad_1_b    
        # w1a = w1a - learning_rate * grad_1_a 

    if iter > 300:
        learning_rate = learning_rate * 0.9
        alpha = alpha * 1.1

    print("Current iter : ",iter, " Current cost: ", total_error,end='\n')
    total_error = 0

print('\n')
predict = np.array([])
for image_index in range(len(image_test_matrix)):
    
    current_image = image_test_matrix[image_index]

    l1aIN,l1bIN = np.pad(current_image,1,mode='constant'),np.pad(current_image,1,mode='constant')

    l1a = convolve2d(l1aIN,w1a,mode='valid')
    l1aA = ReLU(l1a)
    l1aM = skimage.measure.block_reduce(l1aA, block_size=(2,2), func=np.max)

    l1b = convolve2d(l1bIN,w1b,mode='valid')
    l1bA = arctanh(l1b)
    l1bM = skimage.measure.block_reduce(l1bA, block_size=(2,2), func=np.max)

    l2aIN,l2bIN =  np.pad(l1aM,1,mode='constant'),np.pad(l1aM,1,mode='constant')
    l2cIN,l2dIN =  np.pad(l1bM,1,mode='constant'),np.pad(l1bM,1,mode='constant')

    l2a = convolve2d(l2aIN,w1b,mode='valid')
    l2aA = arctanh(l2a)
    l2aM = skimage.measure.block_reduce(l2aA, block_size=(2,2), func=np.max) 

    l2b = convolve2d(l2bIN,w1b,mode='valid')
    l2bA = ReLU(l2b)
    l2bM = skimage.measure.block_reduce(l2bA, block_size=(2,2), func=np.max)

    l2c = convolve2d(l2cIN,w1b,mode='valid')
    l2cA = arctanh(l2c)
    l2cM = skimage.measure.block_reduce(l2cA, block_size=(2,2), func=np.max)

    l2d = convolve2d(l2dIN,w1b,mode='valid')
    l2dA = tanh(l2d)
    l2dM = skimage.measure.block_reduce(l2dA, block_size=(2,2), func=np.max)

    l3IN = np.expand_dims(np.hstack([l2aM.ravel(),l2bM.ravel(),l2cM.ravel(),l2dM.ravel() ]),axis=0)

    l3 = l3IN.dot(w3)
    l3A = arctanh(l3)

    l4 = l3A.dot(w4)
    l4A = tanh(l4)

    l5 = l4A.dot(w5)
    l5A = log(l5)

    predict = np.append(predict,l5A)


print('---- Ground Truth -----')
print(image_test_label.T)

print('---- Predicted  -----')
print(predict.T)

print('---- Predicted Rounded -----')
print(np.round(predict.T))
# -- end code --