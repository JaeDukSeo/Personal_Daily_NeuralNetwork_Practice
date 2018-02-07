import numpy as np,sys
import tensorflow as tf
from sklearn.utils import shuffle
from scipy import signal
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(5678)

np.set_printoptions(precision=3,suppress=True)

def arctan(x):
    return np.tanh(x)
def d_arctan(x):
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
# def d_arctan(x):
#     return 1 / (1 + x ** 2)

def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()


# 1. Read Data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True).test
images,label = shuffle(mnist.images,mnist.labels)

test_image_num,training_image_num = 10,200
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]

num_epoch = 100
learning_rate = 0.05
learning_rate_conv = 0.008
total_cost = 0
batch_size = 2
alpha = 0.001

w1 = np.random.randn(5,5)

w2a = np.random.randn(4,4)
w2b = np.random.randn(4,4)

w3a = np.random.randn(3,3)
w3b = np.random.randn(3,3)
w3c = np.random.randn(3,3)

w4 = np.random.randn(1083,1024)
w5 = np.random.randn(1024,512)
w6 = np.random.randn(512,256) 
w7 = np.random.randn(256,10) * 0.1

v1,v2a,v2b  = 0,0,0
v3a,v3b,v3c = 0,0,0
v4,v5,v6,v7 = 0,0,0,0

for iter in range(num_epoch):
    
    for current_batch_index in range(0,len(training_images),batch_size):
        
        current_image = training_images[current_batch_index:current_batch_index+batch_size,:]
        current_label = training_lables[current_batch_index:current_batch_index+batch_size,:]
        current_image_reshape = np.reshape(current_image,(batch_size,28,28))

        l1 = signal.convolve(current_image_reshape, np.expand_dims(w1,axis=0)  , mode='valid')
        l1A = ReLu(l1)

        l2a  = signal.convolve(l1A, np.expand_dims(w2a,axis=0)  , mode='valid')
        l2Aa = ReLu(l2a)
        l2b  = signal.convolve(l1A, np.expand_dims(w2b,axis=0)  , mode='valid')
        l2Ab = ReLu(l2b)

        l3a  = signal.convolve(l2Aa, np.expand_dims(w3a,axis=0)  , mode='valid')
        l3Aa = ReLu(l3a)
        l3b  = signal.convolve(l2Ab, np.expand_dims(w3b,axis=0)  , mode='valid')
        l3Ab = ReLu(l3b)
        l3c  = signal.convolve(np.divide(np.add(l2Aa,l2Ab),2), np.expand_dims(w3c,axis=0)  , mode='valid')
        l3Ac = ReLu(l3c)

        l4Input = np.hstack((np.reshape(l3Aa,(batch_size,-1)),np.reshape(l3Ab,(batch_size,-1)),np.reshape(l3Ac,(batch_size,-1))))
        l4 = l4Input.dot(w4)
        l4A = arctan(l4)

        l5 = l4A.dot(w5)
        l5A = arctan(l5)

        l6 = l5A.dot(w6)
        l6A = arctan(l6)

        l7 = l6A.dot(w7)
        l7A = arctan(l7)
        l7Soft = softmax(l7A)

        cost = ( -(current_label * np.log(l7Soft) + ( 1-current_label ) * np.log(1 - l7Soft)    )).sum() / batch_size
        print('Update Cost  : ',cost,end='\r')
        total_cost += cost

        grad_7_part_1 = l7Soft - current_label
        grad_7_part_2 = d_arctan(l7)
        grad_7_part_3 = l6A
        grad_7 =    grad_7_part_3.T.dot(grad_7_part_1 * grad_7_part_2)   

        grad_6_part_1 = (grad_7_part_1 * grad_7_part_2).dot(w7.T)
        grad_6_part_2 = d_arctan(l6)
        grad_6_part_3 = l5A
        grad_6 =   grad_6_part_3.T.dot(grad_6_part_1 * grad_6_part_2)

        grad_5_part_1 = (grad_6_part_1 * grad_6_part_2).dot(w6.T)
        grad_5_part_2 = d_arctan(l5)
        grad_5_part_3 = l4A
        grad_5 =   grad_5_part_3.T.dot(grad_5_part_1 * grad_5_part_2)

        grad_4_part_1 = (grad_5_part_1 * grad_5_part_2).dot(w5.T)
        grad_4_part_2 = d_arctan(l4)
        grad_4_part_3 = l4Input
        grad_4 =   grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

        grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)

        grad_3_part_1a = np.reshape(grad_3_part_1[:,:361],(batch_size,19,19))
        grad_3_part_2a = d_ReLu(l3a)
        grad_3_part_3a = l2Aa
        grad_3_a = np.squeeze(np.rot90(signal.convolve( grad_3_part_3a, np.rot90(grad_3_part_1a * grad_3_part_2a ,2)    ,mode='valid') ,2))

        grad_3_part_1b = np.reshape(grad_3_part_1[:,361:361*2],(batch_size,19,19))
        grad_3_part_2b = d_ReLu(l3b)
        grad_3_part_3b = l2Ab
        grad_3_b = np.squeeze(np.rot90(signal.convolve( grad_3_part_3b, np.rot90(grad_3_part_1b * grad_3_part_2b ,2)    ,mode='valid') ,2))

        grad_3_part_1c = np.reshape(grad_3_part_1[:,361*2:361*3],(batch_size,19,19))
        grad_3_part_2c = d_ReLu(l3c)
        grad_3_part_3c = np.divide(np.add(l2Aa,l2Ab),2)
        grad_3_c = np.squeeze(np.rot90(signal.convolve( grad_3_part_3c, np.rot90(grad_3_part_1c * grad_3_part_2c ,2)    ,mode='valid') ,2))
        
        grad_2_part_1 = signal.convolve(np.expand_dims(w3c,axis=0),  np.rot90(np.pad(grad_3_part_1c*grad_3_part_2c*0.5,pad_width=((0,0),(2,2),(2,2))    ,mode='constant') ,2)   ,           mode='valid')

        grad_2_part_1a = signal.convolve(np.expand_dims(w3a,axis=0),  np.rot90(np.pad(grad_3_part_1a*grad_3_part_2a,pad_width=((0,0),(2,2),(2,2))    ,mode='constant')   ,2) ,           mode='valid')
        grad_2_part_2a = d_ReLu(l2a)    
        grad_2_part_3a = l1A
        grad_2_a =  np.squeeze(np.rot90(signal.convolve(grad_2_part_3a, np.rot90(np.pad((grad_2_part_1+grad_2_part_1a)*grad_2_part_2a,pad_width=((0,0),(3,3),(3,3)),mode='constant'      )    ,2)        ,mode='valid')         ,2)   )

        grad_2_part_1b = signal.convolve(np.expand_dims(w3b,axis=0),  np.rot90(np.pad(grad_3_part_1b*grad_3_part_2b,pad_width=((0,0),(2,2),(2,2))    ,mode='constant')   ,2) ,           mode='valid')
        grad_2_part_2b = d_ReLu(l2b)    
        grad_2_part_3b = l1A
        grad_2_b = np.squeeze(np.rot90(signal.convolve(grad_2_part_3b, np.rot90(np.pad((grad_2_part_1+grad_2_part_1b)*grad_2_part_2b,pad_width=((0,0),(3,3),(3,3)),mode='constant'      )    ,2)    ,mode='valid')         ,2) )  

        grad_1_part_1a = signal.convolve(np.expand_dims(w2a,axis=0),  np.rot90(np.pad( (grad_2_part_1+grad_2_part_1a) * grad_2_part_2a  ,pad_width=((0,0),(3,3),(3,3))    ,mode='constant')   ,2) ,        mode='valid')
        grad_1_part_1b = signal.convolve(np.expand_dims(w2b,axis=0),  np.rot90(np.pad( (grad_2_part_1+grad_2_part_1b) * grad_2_part_2b  ,pad_width=((0,0),(3,3),(3,3))    ,mode='constant')   ,2) ,        mode='valid')
        grad_1_part_2 =d_ReLu(l1)
        grad_1_part_3 = current_image_reshape
        grad_1 =  np.squeeze(np.rot90(signal.convolve(grad_1_part_3, np.rot90(np.pad((grad_1_part_1a+grad_1_part_1b)*grad_1_part_2,pad_width=((0,0),(4,4),(4,4)),mode='constant'      )    ,2)    ,mode='valid')         ,2) )  


        v7 = v7 *alpha + learning_rate * grad_7
        v6 = v6 *alpha + learning_rate * grad_6
        v5 = v5 *alpha + learning_rate * grad_5
        v4 = v4 *alpha + learning_rate * grad_4
        
        v3a = v3a *alpha + learning_rate_conv * grad_3_a
        v3b = v3b *alpha + learning_rate_conv * grad_3_b
        v3c = v3c *alpha + learning_rate_conv * grad_3_c

        v2a = v2a *alpha + learning_rate_conv * grad_2_a
        v2b = v2b *alpha + learning_rate_conv * grad_2_b

        v1 = v1 *alpha + learning_rate_conv * grad_1

        w7 = w7 - v7
        w6 = w6 - v6
        w5 = w5 - v5
        w4 = w4 - v4

        w3a = w3a - v3a
        w3b = w3b - v3b
        w3c = w3c - v3c
        
        w2a = w2a - v2a
        w2b = w2b - v2b
        w1 =   w1 - v1

    if iter % 3 == 0 :
        print("current Iter: ", iter, " Current Cost :", total_cost)

        for current_batch_index in range(10):

            testing_images,testing_lables = shuffle(testing_images,testing_lables)

            current_batch = np.expand_dims(testing_images[current_batch_index,:],axis=0)
            current_batch_label = testing_lables[current_batch_index,:]
            current_batch_reshape = np.reshape(current_batch,(28,28))

            l1 = signal.convolve(current_batch_reshape, w1  , mode='valid')
            l1A = ReLu(l1)

            l2a  = signal.convolve(l1A, w2a  , mode='valid')
            l2Aa = ReLu(l2a)
            l2b  = signal.convolve(l1A, w2b  , mode='valid')
            l2Ab = ReLu(l2b)

            l3a  = signal.convolve(l2Aa, w3a  , mode='valid')
            l3Aa = ReLu(l3a)
            l3b  = signal.convolve(l2Ab, w3b  , mode='valid')
            l3Ab = ReLu(l3b)
            l3c  = signal.convolve(np.divide(np.add(l2Aa,l2Ab),2), w3c  , mode='valid')
            l3Ac = ReLu(l3c)

            l4Input = np.hstack((np.reshape(l3Aa,(1,-1)),np.reshape(l3Ab,(1,-1)),np.reshape(l3Ac,(1,-1))))
            l4 = l4Input.dot(w4)
            l4A = arctan(l4)

            l5 = l4A.dot(w5)
            l5A = arctan(l5)

            l6 = l5A.dot(w6)
            l6A = arctan(l6)

            l7 = l6A.dot(w7)
            l7A = arctan(l7)
            l7Soft = softmax(l7A)

            print('Current Predict : ',
            np.where(l7Soft[0] == l7Soft[0].max())[0], 
            "Ground Truth   : ",np.where(current_batch_label == current_batch_label.max())[0] 
            )
        print('------------------------')
    total_cost = 0





# -- end code --