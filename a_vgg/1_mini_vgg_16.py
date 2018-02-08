import numpy as np,sys
import tensorflow as tf
from sklearn.utils import shuffle
from scipy.signal import convolve2d
import skimage.measure
import matplotlib.pyplot as plt
# skimage.measure.block_reduce(l1, (2,2), np.max)
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
num_epoch = 150
learing_rate = 0.01
learing_rate_conv = 0.0001
total_cost = 0
cost_array =[]

w1 = np.random.randn(3,3)
w2 = np.random.randn(3,3)

w3a = np.random.randn(3,3)
w3b = np.random.randn(3,3)

w4a = np.random.randn(3,3)
w4b = np.random.randn(3,3)

w5 = np.random.randn(98,1024)
w6 = np.random.randn(1024,512)
w7 = np.random.randn(512,10) * 0.2


for iter in range(num_epoch):

    for image_index in range(len(training_images)):
        
        current_image = np.expand_dims(training_images[image_index,:],axis=0)
        current_label = np.expand_dims(training_lables[image_index,:],axis=0)
        current_image_reshape = np.reshape(current_image,(28,28))

        l1 = convolve2d(np.pad(current_image_reshape,1,'constant'),w1,'valid')
        l1A = ReLu(l1)

        l2 = convolve2d(np.pad(l1A,1,'constant'),w2,'valid')
        l2A = ReLu(l2)   
        l2Mean = skimage.measure.block_reduce(l2A, (2,2), np.mean)
        
        l3a = convolve2d(np.pad(l2Mean,1,'constant'),w3a,'valid')
        l3Aa = ReLu(l3a)
        l3b = convolve2d(np.pad(l2Mean,1,'constant'),w3b,'valid')
        l3Ab = ReLu(l3b)

        l4a = convolve2d(np.pad(l3Aa,1,'constant'),w4a,'valid')
        l4aA = ReLu(l4a)
        l4b = convolve2d(np.pad(l3Ab,1,'constant'),w4b,'valid')
        l4bA = ReLu(l4b)

        l4aMean = skimage.measure.block_reduce(l4aA, (2,2), np.mean)
        l4bMean = skimage.measure.block_reduce(l4bA, (2,2), np.mean)
        
        l5Input = np.expand_dims(np.hstack(( l4aMean.ravel(), l4bMean.ravel()    )),axis=0)
        l5 = l5Input.dot(w5)   
        l5A = tanh(l5)     

        l6 = l5A.dot(w6)   
        l6A = tanh(l6) 
        
        l7 = l6A.dot(w7)   
        l7A = arctan(l7) 
        l7Soft = softmax(l7A)

        cost = ( -(current_label * np.log(l7Soft) + ( 1-current_label ) * np.log(1 - l7Soft)    )).sum() 
        total_cost += cost
        print("Real Time Update Cost: ", cost,end='\r')

        grad_7_part_1 = l7Soft - current_label
        grad_7_part_2 = d_arctan(l7)
        grad_7_part_3 = l6A
        grad_7 = grad_7_part_3.T.dot(grad_7_part_1 * grad_7_part_2)

        grad_6_part_1 = (grad_7_part_1 * grad_7_part_2).dot(w7.T)
        grad_6_part_2 = d_tanh(l6)
        grad_6_part_3 = l5A
        grad_6 = grad_6_part_3.T.dot(grad_6_part_1 * grad_6_part_2)       

        grad_5_part_1 = (grad_6_part_1 * grad_6_part_2).dot(w6.T)
        grad_5_part_2 = d_tanh(l5)
        grad_5_part_3 = l5Input
        grad_5 = grad_5_part_3.T.dot(grad_5_part_1 * grad_5_part_2)   

        grad_4_Input = (grad_5_part_1 * grad_5_part_2).dot(w5.T)
        grad_4_part_1a = np.reshape(grad_4_Input[:,:49],(7,7)   ).repeat(2,axis=0).repeat(2,axis=1)
        grad_4_part_2a = d_ReLu(l4a)
        grad_4_part_3a = l3Aa       
        grad_4_a = np.rot90(convolve2d(np.pad(grad_4_part_3a,1,'constant'),np.rot90(grad_4_part_1a*grad_4_part_2a,2),'valid'),2)
        
        grad_4_part_1b = np.reshape(grad_4_Input[:,49:],(7,7)   ).repeat(2,axis=0).repeat(2,axis=1)
        grad_4_part_2b = d_ReLu(l4b)
        grad_4_part_3b = l3Ab       
        grad_4_b = np.rot90(convolve2d(np.pad(grad_4_part_3b,1,'constant'),np.rot90(grad_4_part_1b*grad_4_part_2b,2),'valid'),2)

        grad_3_part_1a = convolve2d(w4a,np.rot90( np.pad(grad_4_part_1a  * grad_4_part_2a,1,'constant'),2)    ,'valid')
        grad_3_part_2a = d_ReLu(l3a)
        grad_3_part_3a = l2Mean
        grad_3_a =np.rot90(convolve2d(np.pad(grad_3_part_3a,1,'constant'),np.rot90(grad_3_part_1a*grad_3_part_2a,2),'valid'),2)

        grad_3_part_1b = convolve2d(w4b,np.rot90( np.pad(grad_4_part_1b  * grad_4_part_2b,1,'constant'),2)    ,'valid')
        grad_3_part_2b = d_ReLu(l3b)
        grad_3_part_3b = l2Mean
        grad_3_b =np.rot90(convolve2d(np.pad(grad_3_part_3b,1,'constant'),np.rot90(grad_3_part_1b*grad_3_part_2b,2),'valid'),2)

        grad_2_part_1 = (convolve2d(w3a,np.rot90( np.pad(grad_3_part_1a  * grad_3_part_2a,1,'constant'),2)    ,'valid') +\
                       convolve2d(w3b,np.rot90( np.pad(grad_3_part_1b  * grad_3_part_2b,1,'constant'),2)    ,'valid')).repeat(2,axis=0).repeat(2,axis=1)
        grad_2_part_2 = d_ReLu(l2)
        grad_2_part_3 = l1A
        grad_2 = np.rot90(convolve2d(np.pad(grad_2_part_3,1,'constant'),np.rot90(grad_2_part_1*grad_2_part_2,2),'valid'),2)

        grad_1_part_1 = convolve2d(w2,np.rot90( np.pad(grad_2_part_1  * grad_2_part_2,1,'constant'),2)    ,'valid')
        grad_1_part_2 =  d_ReLu(l1)
        grad_1_part_3 = current_image_reshape
        grad_1 = np.rot90(convolve2d(np.pad(grad_1_part_3,1,'constant'),np.rot90(grad_1_part_1*grad_1_part_2,2),'valid'),2)

        w1 = w1 - learing_rate_conv * grad_1
        w2 = w1 - learing_rate_conv * grad_2

        w3a = w3a - learing_rate_conv * grad_3_a
        w3b = w3b - learing_rate_conv * grad_3_b

        w4a = w4a - learing_rate_conv * grad_4_a
        w4b = w4b - learing_rate_conv * grad_4_b

        w5 = w5 - learing_rate * grad_5
        w6 = w6 - learing_rate * grad_6
        w7 = w7 - learing_rate * grad_7
        


    if iter % 10 == 0 :
        print("current Iter: ", iter, " Current Cost :", total_cost/len(training_images))

        for current_batch_index in range(30):

            testing_images,testing_lables = shuffle(testing_images,testing_lables)

            current_batch = np.expand_dims(testing_images[current_batch_index,:],axis=0)
            current_batch_label = testing_lables[current_batch_index,:]
            current_batch_reshape = np.reshape(current_batch,(28,28))

            l1 = convolve2d(np.pad(current_batch_reshape,1,'constant'),w1,'valid')
            l1A = ReLu(l1)

            l2 = convolve2d(np.pad(l1A,1,'constant'),w2,'valid')
            l2A = ReLu(l2)   
            l2Mean = skimage.measure.block_reduce(l2A, (2,2), np.mean)
            
            l3a = convolve2d(np.pad(l2Mean,1,'constant'),w3a,'valid')
            l3Aa = ReLu(l3a)
            l3b = convolve2d(np.pad(l2Mean,1,'constant'),w3b,'valid')
            l3Ab = ReLu(l3b)

            l4a = convolve2d(np.pad(l3Aa,1,'constant'),w4a,'valid')
            l4aA = ReLu(l4a)
            l4b = convolve2d(np.pad(l3Ab,1,'constant'),w4b,'valid')
            l4bA = ReLu(l4b)

            l4aMean = skimage.measure.block_reduce(l4aA, (2,2), np.mean)
            l4bMean = skimage.measure.block_reduce(l4bA, (2,2), np.mean)
            
            l5Input = np.expand_dims(np.hstack(( l4aMean.ravel(), l4bMean.ravel()    )),axis=0)
            l5 = l5Input.dot(w5)   
            l5A = tanh(l5)     

            l6 = l5A.dot(w6)   
            l6A = tanh(l6) 
            
            l7 = l6A.dot(w7)   
            l7A = arctan(l7) 
            l7Soft = softmax(l7A)

            print('Current Predict : ',
            np.where(l7Soft[0] == l7Soft[0].max())[0], 
            "Ground Truth   : ",np.where(current_batch_label == current_batch_label.max())[0],
            # '\n',
            # l7Soft,'\n',current_batch_label
            )
        print('------------------------')
    cost_array.append(total_cost)
    total_cost = 0










print('=======================================')
print('==============FINAL====================')
correct = 0
for current_batch_index in range(len(testing_images)):

    current_batch = np.expand_dims(testing_images[current_batch_index,:],axis=0)
    current_batch_label = testing_lables[current_batch_index,:]

    current_batch_reshape = np.reshape(current_batch,(28,28))

    l1 = convolve2d(np.pad(current_batch_reshape,1,'constant'),w1,'valid')
    l1A = ReLu(l1)

    l2 = convolve2d(np.pad(l1A,1,'constant'),w2,'valid')
    l2A = ReLu(l2)   
    l2Mean = skimage.measure.block_reduce(l2A, (2,2), np.mean)
    
    l3a = convolve2d(np.pad(l2Mean,1,'constant'),w3a,'valid')
    l3Aa = ReLu(l3a)
    l3b = convolve2d(np.pad(l2Mean,1,'constant'),w3b,'valid')
    l3Ab = ReLu(l3b)

    l4a = convolve2d(np.pad(l3Aa,1,'constant'),w4a,'valid')
    l4aA = ReLu(l4a)
    l4b = convolve2d(np.pad(l3Ab,1,'constant'),w4b,'valid')
    l4bA = ReLu(l4b)

    l4aMean = skimage.measure.block_reduce(l4aA, (2,2), np.mean)
    l4bMean = skimage.measure.block_reduce(l4bA, (2,2), np.mean)
    
    l5Input = np.expand_dims(np.hstack(( l4aMean.ravel(), l4bMean.ravel()    )),axis=0)
    l5 = l5Input.dot(w5)   
    l5A = tanh(l5)     

    l6 = l5A.dot(w6)   
    l6A = tanh(l6) 
    
    l7 = l6A.dot(w7)   
    l7A = arctan(l7) 
    l7Soft = softmax(l7A)

    print(' Current Predict : ',np.where(l7Soft[0] == l7Soft[0].max())[0], " Ground Truth : ", np.where(current_batch_label == current_batch_label.max())[0] )

    if np.where(l7Soft[0] == l7Soft[0].max())[0] == np.where(current_batch_label == current_batch_label.max())[0]:
        correct += 1

print('Correct : ',correct, ' Out of : ',len(testing_images) )
plt.title("Cost over time")
plt.plot(np.arange(num_epoch), cost_array)
plt.show()













# --- end code ---