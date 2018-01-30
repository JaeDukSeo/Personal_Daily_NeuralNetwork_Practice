import numpy as np,sys
from sklearn.datasets import load_digits
from scipy.ndimage.filters import maximum_filter
import skimage.measure
from scipy.signal import convolve2d
from scipy import fftpack
from sklearn.utils import shuffle

np.random.seed(1432432)

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
learning_rate = 0.000008
learning_rate_rec = 0.000000008
learning_rate_fc = 0.1
alpha= 0.00001
total_error = 0

wxg,wxc = np.random.randn(5,5)*0.5,np.random.randn(5,5)*0.5
wrecg,wrecc = np.random.randn(3,3)*0.002,np.random.randn(3,3)*0.002
w_full_1,w_full_2 = np.random.randn(16,34)*0.1,np.random.randn(34,1)*0.1

v1,v2,v3    =0,0,0
v4,v5,v6 = 0,0,0

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

h = np.random.randn(3,4,4)

for iter in range(num_epoch):

    for image_index in range(len(image_matrix)):
        
        current_image = image_matrix[image_index]
        current_image_label = image_label[image_index]

        cg1_h_IN = np.pad(h[0,:,:],1,mode='constant')
        c1 = convolve2d(cg1_h_IN,wrecc,mode='valid') + convolve2d(current_image,wxc,mode='valid') 
        c1A = tanh(c1)
        g1 = convolve2d(cg1_h_IN,wrecg,mode='valid') + convolve2d(current_image,wxg,mode='valid') 
        g1A = arctanh(g1)
        h[1,:,:] = g1A * h[0,:,:] + ( 1- g1A) * c1A
        
        cg2_h_IN = np.pad(h[1,:,:],1,mode='constant')
        c2 = convolve2d(cg2_h_IN,wrecc,mode='valid') + convolve2d(current_image,wxc,mode='valid') 
        c2A = tanh(c2)
        g2 = convolve2d(cg2_h_IN,wrecg,mode='valid') + convolve2d(current_image,wxg,mode='valid') 
        g2A = arctanh(g2)
        h[2,:,:] = g2A * h[1,:,:] + ( 1- g2A) * c2A

        full_layer_1_IN = np.expand_dims(h[2,:,:].ravel(),axis=0)
        full_layer_1 = full_layer_1_IN.dot(w_full_1)
        full_layer_1_A = arctanh(full_layer_1)

        full_layer_2 = full_layer_1_A.dot(w_full_2)
        full_layer_2_A = log(full_layer_2)   

        cost = np.square(full_layer_2_A-current_image_label).sum() * 0.5
        total_error = total_error + cost

        grad_3_part_1 = full_layer_2_A-current_image_label
        grad_3_part_2 = d_log(full_layer_2)
        grad_3_part_3 = full_layer_1_A
        grad_3 =    grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2) 

        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w_full_2.T)
        grad_2_part_2 = d_arctan(full_layer_1)
        grad_2_part_3 = full_layer_1_IN
        grad_2 =    grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2) 

        grad_ts3_IN = np.reshape((grad_2_part_1 * grad_2_part_2).dot(w_full_1.T),(4,4))

        grad_ts2_wrecc_part_1 = ( 1- g2A) * grad_ts3_IN
        grad_ts2_wrecc_part_2 = d_arctan(c2)
        grad_ts2_wrecc_part_3 = cg2_h_IN
        grad_ts2_wrecc = np.rot90(
            convolve2d(grad_ts2_wrecc_part_3,
                np.rot90(grad_ts2_wrecc_part_1 * grad_ts2_wrecc_part_2,2)
                ,mode='valid'),2
            )

        grad_ts2_wxc_part_1 = ( 1- g2A) * grad_ts3_IN
        grad_ts2_wxc_part_2 = d_arctan(c2)
        grad_ts2_wxc_part_3 = current_image
        grad_ts2_wxc = np.rot90(
            convolve2d(grad_ts2_wxc_part_3,
                np.rot90(grad_ts2_wxc_part_1 * grad_ts2_wxc_part_2,2)
                ,mode='valid'),2
            )

        grad_ts2_wrecg_part_1 = (h[1,:,:] - c2A) * grad_ts3_IN
        grad_ts2_wrecg_part_2 = d_arctan(g2)
        grad_ts2_wrecg_part_3 = cg2_h_IN
        grad_ts2_wrecg = np.rot90(
            convolve2d(grad_ts2_wrecg_part_3,
                np.rot90(grad_ts2_wrecg_part_1 * grad_ts2_wrecg_part_2,2)
                ,mode='valid'),2
            )

        grad_ts2_wxg_part_1 = (h[1,:,:] - c2A) * grad_ts3_IN
        grad_ts2_wxg_part_2 = d_arctan(g2)
        grad_ts2_wxg_part_3 = current_image
        grad_ts2_wxg = np.rot90(
            convolve2d(grad_ts2_wxg_part_3,
                np.rot90(grad_ts2_wxg_part_1 * grad_ts2_wxg_part_2,2)
                ,mode='valid'),2
            )

        grad_ts1_wrecc_part_1 = (
            convolve2d(np.pad(wrecc,2,mode='constant'),np.rot90(grad_ts2_wrecc_part_1 * grad_ts2_wrecc_part_2,2),mode='valid')  + \
            g2A + \
            convolve2d(np.pad(wrecg,2,mode='constant'),np.rot90(grad_ts2_wrecg_part_1 * grad_ts2_wrecg_part_2,2),mode='valid')  
        ) * ( 1- g2A)
        grad_ts1_wrecc_part_2 = d_arctan(c1)
        grad_ts1_wrecc_part_3 = cg1_h_IN
        grad_ts1_wrecc = np.rot90(
            convolve2d(grad_ts1_wrecc_part_3,
                np.rot90(grad_ts1_wrecc_part_1 * grad_ts1_wrecc_part_2,2)
                ,mode='valid'),2
            )

        grad_ts1_wxc_part_1 = (
            convolve2d(np.pad(wrecc,2,mode='constant'),np.rot90(grad_ts2_wrecc_part_1 * grad_ts2_wrecc_part_2,2),mode='valid')  + \
            g2A + \
            convolve2d(np.pad(wrecg,2,mode='constant'),np.rot90(grad_ts2_wrecg_part_1 * grad_ts2_wrecg_part_2,2),mode='valid')  
        ) * ( 1- g2A)
        grad_ts1_wxc_part_2 = d_arctan(c1)
        grad_ts1_wxc_part_3 = current_image
        grad_ts1_wxc = np.rot90(
            convolve2d(grad_ts1_wxc_part_3,
                np.rot90(grad_ts1_wxc_part_1 * grad_ts1_wxc_part_2,2)
                ,mode='valid'),2
            )

        grad_ts1_wrecg_part_1 = (
            convolve2d(np.pad(wrecc,2,mode='constant'),np.rot90(grad_ts2_wrecc_part_1 * grad_ts2_wrecc_part_2,2),mode='valid')   + \
            g2A + \
            convolve2d(np.pad(wrecg,2,mode='constant'),np.rot90(grad_ts2_wrecg_part_1 * grad_ts2_wrecg_part_2,2),mode='valid')
        ) * (h[0,:,:] * c1A) 
        grad_ts1_wrecg_part_2 = d_arctan(g1)
        grad_ts1_wrecg_part_3 = cg1_h_IN
        grad_ts1_wrecg = np.rot90(
            convolve2d(grad_ts1_wrecg_part_3,
                np.rot90(grad_ts1_wrecg_part_1 * grad_ts1_wrecg_part_2,2)
                ,mode='valid'),2
            )

        grad_ts1_wxg_part_1 = (
            convolve2d(np.pad(wrecc,2,mode='constant'),np.rot90(grad_ts2_wrecc_part_1 * grad_ts2_wrecc_part_2,2),mode='valid')   + \
            g2A + \
            convolve2d(np.pad(wrecg,2,mode='constant'),np.rot90(grad_ts2_wrecg_part_1 * grad_ts2_wrecg_part_2,2),mode='valid')
        ) * (h[0,:,:] * c1A) 
        grad_ts1_wxg_part_2 = d_arctan(g1)
        grad_ts1_wxg_part_3 = current_image
        grad_ts1_wxg = np.rot90(
            convolve2d(grad_ts1_wxg_part_3,
                np.rot90(grad_ts1_wxg_part_1 * grad_ts1_wxg_part_2,2)
                ,mode='valid'),2
            )
        
        v1 = v1 * alpha + learning_rate_rec * (grad_ts1_wrecc + grad_ts2_wrecc) 
        v2 = v2 * alpha + learning_rate * (grad_ts1_wxc + grad_ts2_wxc )
        wrecc = wrecc - v1
        wxc = wxc - v2

        v3 = v3 * alpha + learning_rate_rec * (grad_ts1_wrecg + grad_ts2_wrecg)
        v4 = v4 * alpha + learning_rate * (grad_ts1_wxg + grad_ts2_wxg)
        wrecg = wrecg - v3
        wxg = wxg - v4
        
        v5 = v5 * alpha + learning_rate_fc * grad_2 
        v6 = v6 * alpha + learning_rate_fc * grad_3 
        w_full_1 = w_full_1 - v5
        w_full_2 = w_full_2 - v6

    print("current Iter: ",iter, " Current cost: ",total_error,end='\r')
    total_error = 0

    # if iter > 80:
    #     alpha = alpha * 0.1

print('\n')
predict = np.array([])
for image_index in range(len(image_test_matrix)):
    
    current_image = image_test_matrix[image_index]
    cg1_h_IN = np.pad(h[0,:,:],1,mode='constant')
    c1 = convolve2d(cg1_h_IN,wrecc,mode='valid') + convolve2d(current_image,wxc,mode='valid') 
    c1A = tanh(c1)
    g1 = convolve2d(cg1_h_IN,wrecg,mode='valid') + convolve2d(current_image,wxg,mode='valid') 
    g1A = arctanh(g1)
    h[1,:,:] = g1A * h[0,:,:] + ( 1- g1A) * c1A
    
    cg2_h_IN = np.pad(h[1,:,:],1,mode='constant')
    c2 = convolve2d(cg2_h_IN,wrecc,mode='valid') + convolve2d(current_image,wxc,mode='valid') 
    c2A = tanh(c2)
    g2 = convolve2d(cg2_h_IN,wrecg,mode='valid') + convolve2d(current_image,wxg,mode='valid') 
    g2A = arctanh(g2)
    h[2,:,:] = g2A * h[1,:,:] + ( 1- g2A) * c2A

    full_layer_1_IN = np.expand_dims(h[2,:,:].ravel(),axis=0)
    full_layer_1 = full_layer_1_IN.dot(w_full_1)
    full_layer_1_A = arctanh(full_layer_1)

    full_layer_2 = full_layer_1_A.dot(w_full_2)
    full_layer_2_A = log(full_layer_2)   

    predict = np.append(predict,full_layer_2_A)


print('---- Ground Truth -----')
print(image_test_label.T)

print('---- Predicted  -----')
print(predict.T)

print('---- Predicted Rounded -----')
print(np.round(predict.T))
# -- end code --