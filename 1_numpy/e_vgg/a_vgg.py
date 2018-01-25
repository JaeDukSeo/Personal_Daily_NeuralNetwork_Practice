import numpy as np,sys
# Func: Only for 2D convolution 
from scipy.signal import convolve2d

# Func: Only For Back propagation on Max Pooling
from scipy.ndimage.filters import maximum_filter
import skimage.measure

# Func: Import Data
from sklearn.datasets import load_digits

np.random.seed(2678)

def ReLU(x):
    mask  = (x >0) * 1.0 
    return mask * x
def d_ReLU(x):
    mask  = (x >0) * 1.0 
    return mask 

def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))

# 1. Prepare Data
data =load_digits()
image = data.images
label = data.target
num_epoch = 100
learning_rate = 0.1

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


# L1
w1,w2 = np.random.randn(3,3),np.random.randn(3,3)
# L2
w3,w4 = np.random.randn(3,3),np.random.randn(3,3)

# L3
w5,w6,w7,w8 = np.random.randn(3,3),np.random.randn(3,3),np.random.randn(3,3),np.random.randn(3,3)
# L4
w9,w10,w11,w12 = np.random.randn(3,3),np.random.randn(3,3),np.random.randn(3,3),np.random.randn(3,3)

# L5
w13,w14,w15,w16 = np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2)
w17,w18,w19,w20 = np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2)
# L6
w21,w22,w23,w24 = np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2)
w25,w26,w27,w28 = np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2)
# L7
w29,w30,w31,w32 = np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2)
w33,w34,w35,w36 = np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2),np.random.randn(2,2)

# L8
w37_44 = np.random.randn(8,1)
w45_52 = np.random.randn(8,1)
# L9
w53_68 = np.random.randn(16,1)
# L10
w69_84 = np.random.randn(16,1)

# L11
w85_100 = np.random.randn(16,1)
# L12
w101_116 = np.random.randn(16,1)
# L13
w117_132 = np.random.randn(16,1)

# L14
w133 = np.random.randn(16,128)
# L15
w134 = np.random.randn(128,128)
# L16
w135 = np.random.randn(128,1)

for iter in range(num_epoch):
    
    for current_image_index in range(len(image_matrix)):
        
        current_image = image_matrix[current_image_index]
        current_image_label = image_label[current_image_index]

        # 1. Layer 1 and activation function
        l1_1 = convolve2d(current_image,w1,mode='same',boundary='fill',fillvalue=0)
        l1_2 = convolve2d(current_image,w2,mode='same',boundary='fill',fillvalue=0)
        l1_1A,l1_2A = ReLU(l1_1),ReLU(l1_2)

        # 2. Layer 2 and activation function
        l2_1 = convolve2d(l1_1A,w3,mode='same',boundary='fill',fillvalue=0)
        l2_2 = convolve2d(l1_2A,w4,mode='same',boundary='fill',fillvalue=0)
        l2_1A,l2_2A = ReLU(l2_1),ReLU(l2_2)

        # 3. Layer 3 and activation function and input data max pooling
        l3_in_1,l3_in_2 = skimage.measure.block_reduce(l2_1A, block_size=(2,2), func=np.max),skimage.measure.block_reduce(l2_2A, block_size=(2,2), func=np.max)

        l3_1,l3_2 = convolve2d(l3_in_1,w5,mode='same',boundary='fill',fillvalue=0),convolve2d(l3_in_1,w6,mode='same',boundary='fill',fillvalue=0)
        l3_3,l3_4 = convolve2d(l3_in_2,w7,mode='same',boundary='fill',fillvalue=0),convolve2d(l3_in_2,w8,mode='same',boundary='fill',fillvalue=0)
        l3_1A,l3_2A,l3_3A,l3_4A = ReLU(l3_1),ReLU(l3_2),ReLU(l3_3),ReLU(l3_4)

        # 4. Layer 4 and activation function
        l4_1,l4_2 = convolve2d(l3_1A,w9,mode='same',boundary='fill',fillvalue=0),convolve2d(l3_2A,w10,mode='same',boundary='fill',fillvalue=0)
        l4_3,l4_4 = convolve2d(l3_3A,w11,mode='same',boundary='fill',fillvalue=0),convolve2d(l3_4A,w12,mode='same',boundary='fill',fillvalue=0)
        l4_1A,l4_2A,l4_3A,l4_4A = ReLU(l4_1),ReLU(l4_2),ReLU(l4_3),ReLU(l4_4)

        # 5. Layer 5 and activation function and input data max pooling
        l5_in_1,l5_in_2 = skimage.measure.block_reduce(l4_1A, block_size=(2,2), func=np.max),skimage.measure.block_reduce(l4_2A, block_size=(2,2), func=np.max)
        l5_in_3,l5_in_4 = skimage.measure.block_reduce(l4_3A, block_size=(2,2), func=np.max),skimage.measure.block_reduce(l4_4A, block_size=(2,2), func=np.max)

        l5_1,l5_2 = convolve2d(l5_in_1,w13,mode='same',boundary='fill',fillvalue=0),convolve2d(l5_in_1,w14,mode='same',boundary='fill',fillvalue=0)
        l5_3,l5_4 = convolve2d(l5_in_2,w15,mode='same',boundary='fill',fillvalue=0),convolve2d(l5_in_2,w16,mode='same',boundary='fill',fillvalue=0)
        l5_5,l5_6 = convolve2d(l5_in_3,w17,mode='same',boundary='fill',fillvalue=0),convolve2d(l5_in_3,w18,mode='same',boundary='fill',fillvalue=0)
        l5_7,l5_8 = convolve2d(l5_in_4,w19,mode='same',boundary='fill',fillvalue=0),convolve2d(l5_in_4,w20,mode='same',boundary='fill',fillvalue=0)
        l5_1A,l5_2A,l5_3A,l5_4A = ReLU(l5_1),ReLU(l5_2),ReLU(l5_3),ReLU(l5_4)
        l5_5A,l5_6A,l5_7A,l5_8A = ReLU(l5_5),ReLU(l5_6),ReLU(l5_7),ReLU(l5_8)
        
        # 6. Layer 6 and activation function
        l6_1,l6_2 = convolve2d(l5_1A,w21,mode='same',boundary='fill',fillvalue=0),convolve2d(l5_2A,w22,mode='same',boundary='fill',fillvalue=0)
        l6_3,l6_4 = convolve2d(l5_3A,w23,mode='same',boundary='fill',fillvalue=0),convolve2d(l5_4A,w24,mode='same',boundary='fill',fillvalue=0)
        l6_5,l6_6 = convolve2d(l5_5A,w25,mode='same',boundary='fill',fillvalue=0),convolve2d(l5_6A,w26,mode='same',boundary='fill',fillvalue=0)
        l6_7,l6_8 = convolve2d(l5_7A,w27,mode='same',boundary='fill',fillvalue=0),convolve2d(l5_8A,w28,mode='same',boundary='fill',fillvalue=0)
        l6_1A,l6_2A,l6_3A,l6_4A = ReLU(l6_1),ReLU(l6_2),ReLU(l6_3),ReLU(l6_4)
        l6_5A,l6_6A,l6_7A,l6_8A = ReLU(l6_5),ReLU(l6_6),ReLU(l6_7),ReLU(l6_8)

        # 7. Layer 7 and activation function
        l7_1,l7_2 = convolve2d(l6_1A,w29,mode='same',boundary='fill',fillvalue=0),convolve2d(l6_2A,w30,mode='same',boundary='fill',fillvalue=0)
        l7_3,l7_4 = convolve2d(l6_3A,w31,mode='same',boundary='fill',fillvalue=0),convolve2d(l6_4A,w32,mode='same',boundary='fill',fillvalue=0)
        l7_5,l7_6 = convolve2d(l6_5A,w33,mode='same',boundary='fill',fillvalue=0),convolve2d(l6_6A,w34,mode='same',boundary='fill',fillvalue=0)
        l7_7,l7_8 = convolve2d(l6_7A,w35,mode='same',boundary='fill',fillvalue=0),convolve2d(l6_8A,w36,mode='same',boundary='fill',fillvalue=0)
        l7_1A,l7_2A,l7_3A,l7_4A = ReLU(l7_1),ReLU(l7_2),ReLU(l7_3),ReLU(l7_4)
        l7_5A,l7_6A,l7_7A,l7_8A = ReLU(l7_5),ReLU(l7_6),ReLU(l7_7),ReLU(l7_8)

        # 8. Layer 8 and activation function and input data max pooling
        l8_in_1,l8_in_2 = skimage.measure.block_reduce(l7_1A, block_size=(2,2), func=np.max),skimage.measure.block_reduce(l7_2A, block_size=(2,2), func=np.max)
        l8_in_3,l8_in_4 = skimage.measure.block_reduce(l7_3A, block_size=(2,2), func=np.max),skimage.measure.block_reduce(l7_4A, block_size=(2,2), func=np.max)
        l8_in_5,l8_in_6 = skimage.measure.block_reduce(l7_5A, block_size=(2,2), func=np.max),skimage.measure.block_reduce(l7_6A, block_size=(2,2), func=np.max)
        l8_in_7,l8_in_8 = skimage.measure.block_reduce(l7_7A, block_size=(2,2), func=np.max),skimage.measure.block_reduce(l7_8A, block_size=(2,2), func=np.max)
        
        # Vecotrize since it will be too long
        l8_vec = np.expand_dims(np.hstack((l8_in_1.ravel(),l8_in_2.ravel(),
                                           l8_in_3.ravel(),l8_in_4.ravel(),
                                           l8_in_5.ravel(),l8_in_6.ravel(),
                                           l8_in_7.ravel(),l8_in_8.ravel() )),axis=1)
        l8_1,l8_2 = l8_vec*w37_44,l8_vec*w45_52
        l8_1A,l8_2A = ReLU(l8_1),ReLU(l8_2)

        # 9. Layer 9 and activation function
        l9_in = np.expand_dims(np.hstack((l8_1A.ravel(),l8_2A.ravel())),axis=1)

        l9_1 = l9_in * w53_68
        l9_1A= ReLU(l9_1)
        
        # 10. Layer 10 and activation function
        l10_1 = l9_1A * w69_84
        l10_1A= ReLU(l10_1)

        # 11. Layer 11 and activation funciton input data max pooling
        l11_1 = l10_1A * w85_100
        l11_1A= ReLU(l11_1)

        # 12. Layer 11 and activation funciton
        l12_1 = l11_1A * w101_116
        l12_1A= ReLU(l12_1)

        # 13. Layer 11 and activation funciton
        l13_1 = l12_1A * w117_132
        l13_1A= ReLU(l13_1)

        # 14 Fully connected layer 1
        l14_in = l13_1.T
        l14_1 = l14_in.dot(w133)
        l14_1A = ReLU(l14_1)
        
        # 15 Fully connected layer 2
        l15_1 = l14_1A.dot(w134)
        l15_1A = ReLU(l15_1)

        # 16 Fully connected layer 3
        l16_1 = l15_1A.dot(w135)
        l16_1A = log(l16_1)

        cost = np.square(l16_1A - current_image_label).sum() * 0.5
        # if iter % 100 == 0:
        print(" Current Iter : ", iter, " current image index : ", current_image_index, " current cost : ", cost)

        grad_16_part_1 = l16_1A - current_image_label
        grad_16_part_2 = d_log(l16_1)
        grad_16_part_3 = l15_1A
        grad_16 = grad_16_part_3.T.dot(grad_16_part_1 * grad_16_part_2)
        # w135 = w135 - learning_rate * grad_16

        grad_15_part_1 = (grad_16_part_1 * grad_16_part_2).dot(w135.T)  
        grad_15_part_2 = d_ReLU(l15_1)
        grad_15_part_3 = l14_1A
        grad_15 = grad_15_part_3.T.dot(grad_15_part_1 * grad_15_part_2)
        # w134 = w134 - learning_rate * grad_15

        grad_14_part_1 = (grad_15_part_1 * grad_15_part_2).dot(w134.T)  
        grad_14_part_2 = d_ReLU(l14_1)
        grad_14_part_3 = l13_1.T
        grad_14 =  grad_14_part_3.T.dot(grad_14_part_1 * grad_14_part_2)   
        # w133 = w133 - learning_rate * grad_14

        grad_13_part_1 = (grad_14_part_1 * grad_14_part_2).dot(w133.T).T
        grad_13_part_2 = d_ReLU(l13_1)
        grad_13_part_3 = l12_1A
        grad_13 =     grad_13_part_3 * (grad_13_part_1 * grad_13_part_2)   
        # w117_132 = w117_132 - learning_rate * grad_13

        grad_12_part_1 = (grad_13_part_1 * grad_13_part_2) * w117_132  
        grad_12_part_2 = d_ReLU(l12_1)
        grad_12_part_3 = l11_1A
        grad_12 =     grad_12_part_3 * (grad_12_part_1 * grad_12_part_2)  
        # w101_116 = w101_116 - learning_rate * grad_12

        grad_11_part_1 = (grad_12_part_1 * grad_12_part_2) * w101_116
        grad_11_part_2 = d_ReLU(l11_1)
        grad_11_part_3 = l10_1A
        grad_11 =     grad_11_part_3 * grad_11_part_1 * grad_11_part_2
        # w85_100 = w85_100 - learning_rate * grad_11

        grad_10_part_1 = (grad_11_part_1 * grad_11_part_2) * w85_100
        grad_10_part_2 = d_ReLU(l10_1)
        grad_10_part_3 = l9_1A
        grad_10 =     grad_10_part_3 * (grad_10_part_1 * grad_10_part_2)  
        # w69_84 = w69_84 - learning_rate * grad_10

        grad_9_part_1 = (grad_10_part_1 * grad_10_part_2) * w69_84
        grad_9_part_2 = d_ReLU(l9_1)
        grad_9_part_3 = l9_in
        grad_9 =     grad_9_part_3 * (grad_9_part_1 * grad_9_part_2) 
        # w53_68 = w53_68 - learning_rate * grad_9

        grad_8_part_1_a = ((grad_9_part_1 * grad_9_part_2) * w53_68)[:8]
        grad_8_part_1_b = ((grad_9_part_1 * grad_9_part_2) * w53_68)[8:]
        
        grad_8_part_2_a = d_ReLU(l8_1)
        grad_8_part_2_b = d_ReLU(l8_2)

        grad_8_part_3_a = l8_vec
        grad_8_part_3_b = l8_vec

        grad_8_a = grad_8_part_3_a * (grad_8_part_1_a * grad_8_part_2_a )
        grad_8_b = grad_8_part_3_b * (grad_8_part_1_b * grad_8_part_2_b )
        
        # w37_44 = w37_44 - learning_rate * grad_8_a
        # w45_52 = w45_52 - learning_rate * grad_8_b

        grad_7_part_1_a = (grad_8_part_1_a * grad_8_part_2_a) * w37_44
        grad_7_part_1_b = (grad_8_part_1_b * grad_8_part_2_b) * w45_52

        print(grad_7_part_1_a.shape)
        print(grad_7_part_1_b.shape)
        
        


        sys.exit()

        # update weights
        w135 = w135 - learning_rate * grad_16
        w134 = w134 - learning_rate * grad_15
        w133 = w133 - learning_rate * grad_14
        w117_132 = w117_132 - learning_rate * grad_13
        w101_116 = w101_116 - learning_rate * grad_12
        w85_100 = w85_100 - learning_rate * grad_11
        w69_84 = w69_84 - learning_rate * grad_10
        w53_68 = w53_68 - learning_rate * grad_9
        w37_44 = w37_44 - learning_rate * grad_8_a
        w45_52 = w45_52 - learning_rate * grad_8_b

    print(iter)
          


#  -- end code --