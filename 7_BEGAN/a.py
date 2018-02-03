import numpy as np,sys
from scipy.signal import convolve2d
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

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
images, labels = shuffle(temp.images, temp.labels)  
num_epoch = 1
learning_rate = 0.001
value  = 0.002

# 1. Declare Weights Only for Discriminator 
# -------- Encoding Part ----------
D_encode_w0 = np.random.randn(3,3) * value
D_encode_w1 = np.random.randn(3,3)* value

D_encode_w2a = np.random.randn(3,3)* value
D_encode_w2b = np.random.randn(3,3)* value

D_encode_w3a = np.random.randn(3,3)* value
D_encode_w3b = np.random.randn(3,3)* value

D_encode_w4a = np.random.randn(3,3)* value
D_encode_w4b = np.random.randn(3,3)* value
D_encode_w4c = np.random.randn(3,3)* value

D_encode_w5a = np.random.randn(3,3)* value
D_encode_w5b = np.random.randn(3,3)* value
D_encode_w5c = np.random.randn(3,3)* value

D_encode_w6a = np.random.randn(3,3)* value
D_encode_w6b = np.random.randn(3,3)* value
D_encode_w6c = np.random.randn(3,3)* value

# -------- Transition Part ----------
D_trans_wh_a = np.random.randn(147,128) * value
D_trans_wh_b = np.random.randn(128,49) * value

# -------- Deconding Part ----------
D_decode_w7 = np.random.randn(3,3)* value
D_decode_w8 = np.random.randn(3,3)* value

D_decode_w9_a = np.random.randn(3,3)* value
D_decode_w9_b = np.random.randn(3,3)* value
D_decode_w10 = np.random.randn(3,3)* value

D_decode_w11_a = np.random.randn(3,3)* value
D_decode_w11_b = np.random.randn(3,3)* value
D_decode_w12 = np.random.randn(3,3)* value

D_decode_w13 = np.random.randn(3,3)* value


for iter in range(num_epoch):
    
    for current_image in images:

        image_reshape = np.reshape(current_image,(28,28))

        D_0Input = np.pad(image_reshape,1,mode='constant')
        D_0l = convolve2d(D_0Input,D_encode_w0,mode='valid')
        D_l0A = elu(D_0l)

        # ----------------------------
        D_1Input = np.pad(D_l0A,1,mode='constant')
        D_1l = convolve2d(D_1Input,D_encode_w1,mode='valid')
        D_1lA = elu(D_1l)

        D_2Input = np.pad(D_1lA,1,mode='constant')
        D_2l_a = convolve2d(D_2Input,D_encode_w2a,mode='valid')
        D_2lA_a = elu(D_2l_a)
        D_2l_Down_a = block_reduce(D_2lA_a, (2,2), np.mean)
        D_2l_b = convolve2d(D_2Input,D_encode_w2b,mode='valid')
        D_2lA_b = elu(D_2l_b)
        D_2l_Down_b = block_reduce(D_2lA_b, (2,2), np.mean)
        # ----------------------------

        # ----------------------------
        D_3_Input_a = np.pad(D_2l_Down_a,1,mode='constant')
        D_3_Input_b = np.pad(D_2l_Down_b,1,mode='constant')       
        D_3l_a = convolve2d(D_3_Input_a,D_encode_w3a,mode='valid')
        D_3lA_a = elu(D_3l_a)
        D_3l_b = convolve2d(D_3_Input_b,D_encode_w3b,mode='valid')
        D_3lA_b = elu(D_3l_b)    

        D_4_Input_a = np.pad(D_3lA_a,1,mode='constant')
        D_4_Input_b = np.pad(D_3lA_b,1,mode='constant')              
        D_4_Input_c = np.pad(np.mean(D_3lA_a + D_3lA_b),1,mode='constant')              
        D_4l_a = convolve2d(D_3_Input_a,D_encode_w4a,mode='valid')
        D_4lA_a = elu(D_4l_a)
        D_4l_Down_a = block_reduce(D_4lA_a, (2,2), np.mean)
        D_4l_b = convolve2d(D_3_Input_b,D_encode_w4b,mode='valid')
        D_4lA_b = elu(D_4l_b)   
        D_4l_Down_b = block_reduce(D_4lA_b, (2,2), np.mean)
        D_4l_c = convolve2d(D_3_Input_b,D_encode_w4c,mode='valid')
        D_4lA_c = elu(D_4l_c)    
        D_4l_Down_c = block_reduce(D_4lA_c, (2,2), np.mean)
        # ----------------------------

        # ----------------------------
        D_5_Input_a = np.pad(D_4l_Down_a,1,mode='constant')
        D_5_Input_b = np.pad(D_4l_Down_b,1,mode='constant')         
        D_5_Input_c = np.pad(D_4l_Down_c,1,mode='constant')   
        D_5l_a = convolve2d(D_5_Input_a,D_encode_w5a,mode='valid')
        D_5lA_a = elu(D_5l_a)
        D_5l_b = convolve2d(D_5_Input_b,D_encode_w5b,mode='valid')
        D_5lA_b = elu(D_5l_b)   
        D_5l_c = convolve2d(D_5_Input_c,D_encode_w5c,mode='valid')
        D_5lA_c = elu(D_5l_c)    

        D_6_Input_a = np.pad(D_5lA_a,1,mode='constant')
        D_6_Input_b = np.pad(D_5lA_b,1,mode='constant')         
        D_6_Input_c = np.pad(D_5lA_c,1,mode='constant')   
        D_6l_a = convolve2d(D_6_Input_a,D_encode_w6a,mode='valid')
        D_6lA_a = elu(D_6l_a)
        D_6l_b = convolve2d(D_6_Input_b,D_encode_w6b,mode='valid')
        D_6lA_b = elu(D_6l_b)   
        D_6l_c = convolve2d(D_6_Input_c,D_encode_w6c,mode='valid')
        D_6lA_c = elu(D_6l_c)    
        # ----------------------------

        # ----------------------------
        D_h_Input = np.hstack((D_6lA_a.ravel(),D_6lA_b.ravel(),D_6lA_c.ravel()))
        D_wh_l = D_h_Input.dot(D_trans_wh_a)
        D_decode_Input = D_wh_l.dot(D_trans_wh_b)
        # ----------------------------

        # ----------------------------
        D_decode_reshape = np.reshape(D_decode_Input,(7,7) )   
        D_decode_7_Input = np.pad(D_decode_reshape,1,mode='constant')
        D_decond_7 = convolve2d(D_decode_7_Input,D_decode_w7,mode='valid')
        D_decode_8_Input = np.pad(D_decond_7,1,mode='constant')
        D_decond_8 = convolve2d(D_decode_8_Input,D_decode_w7,mode='valid')
        
        # ----------------------------
        D_decode_9_upscale = D_decond_8.repeat(2, axis=0).repeat(2, axis=1)
        D_decode_9_skip_upscale = D_decode_reshape.repeat(2, axis=0).repeat(2, axis=1)
        D_decode_9_Input = np.pad(D_decode_9_upscale,1,mode='constant')
        D_decode_9_skip_Input = np.pad(D_decode_9_skip_upscale,1,mode='constant')
        D_decode_9_a = convolve2d(D_decode_9_Input,D_decode_w9_a,mode='valid')
        D_decode_9_b = convolve2d(D_decode_9_skip_Input,D_decode_w9_b,mode='valid')
        
        D_decode_10_sum = np.add(D_decode_9_a,D_decode_9_b)/2
        D_decode_10_Input = np.pad(D_decode_10_sum,1,mode='constant')
        D_decode_10 = convolve2d(D_decode_10_Input,D_decode_w10,mode='valid')
        # ----------------------------

        # ----------------------------
        D_decode_11_upscale = D_decode_10.repeat(2, axis=0).repeat(2, axis=1)
        D_decode_11_skip_upscale = D_decode_reshape.repeat(4, axis=0).repeat(4, axis=1)
        D_decode_11_Input = np.pad(D_decode_11_upscale,1,mode='constant')
        D_decode_11_skip_Input = np.pad(D_decode_11_skip_upscale,1,mode='constant')
        D_decode_11_a = convolve2d(D_decode_11_Input,D_decode_w11_a,mode='valid')
        D_decode_11_b = convolve2d(D_decode_11_skip_Input,D_decode_w11_b,mode='valid')

        D_decode_12_sum = np.add(D_decode_11_a,D_decode_11_b)/2
        D_decode_12_Input = np.pad(D_decode_12_sum,1,mode='constant')
        D_decode_12 = convolve2d(D_decode_12_Input,D_decode_w12,mode='valid')
        # ----------------------------

        # ----------------------------
        D_decode_13_Input = np.pad(D_decode_12,1,mode='constant')
        D_decode_13 = convolve2d(D_decode_13_Input,D_decode_w13,mode='valid')
        # ----------------------------
        

        print(D_decode_13.shape)
        
        plt.imshow(D_decode_13,cmap='gray')
        plt.show()



        sys.exit()

# -- end code --