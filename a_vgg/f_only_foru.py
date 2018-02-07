import numpy as np,sys
import tensorflow as tf
from sklearn.utils import shuffle
from scipy import signal
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(5678)

np.set_printoptions(precision=3,suppress=True)

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
    exp = np.exp(x)
    return exp/exp.sum()

# 0. Declare Training Data and Labels
mnist_index = input_data.read_data_sets("../MNIST_data/", one_hot=False).test.labels
mnist= input_data.read_data_sets("../MNIST_data/", one_hot=True)
only_to_four = np.where(mnist_index==0)[0],np.where(mnist_index==1)[0],np.where(mnist_index==2)[0],np.where(mnist_index==3)[0],np.where(mnist_index==4)[0]


train = mnist.test
images, labels = train.images, train.labels

only_zero_image,only_zero_label = images[[only_to_four[0]]],    labels[[only_to_four[0]]][:,:5]
only_one_image,only_one_label   = images[[only_to_four[1]]],    labels[[only_to_four[1]]][:,:5]
only_two_image,only_two_label   = images[[only_to_four[2]]],    labels[[only_to_four[2]]][:,:5]
only_three_image,only_three_label = images[[only_to_four[3]]],  labels[[only_to_four[3]]][:,:5]
only_four_image,only_four_label   = images[[only_to_four[4]]],  labels[[only_to_four[4]]][:,:5]

images = np.vstack((only_zero_image,only_one_image,only_two_image,only_three_image,only_four_image))
labels = np.vstack((only_zero_label,only_one_label,only_two_label,only_three_label,only_four_label))
# images = np.vstack((only_zero_image,only_one_image))
# labels = np.vstack((only_zero_label,only_one_label))
images,label = shuffle(images,labels)


test_image_num,training_image_num = 20,600
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]

















# ------------- Weights ---------------
w1 = np.random.randn(5,5)
w2 = np.random.randn(4,4)
w3 = np.random.randn(3,3)

w4 = np.random.randn(361,512)* 0.2
w5 = np.random.randn(512,256)* 0.3
w6 = np.random.randn(256,5)* 0.2

v1,v2,v3,v4,v5,v6 = 0,0,0,0,0,0
m1,m2,m3,m4,m5,m6 = 0,0,0,0,0,0
beta1,beta2,adam_e= 0.9,0.999,0.00000001
learning_rate = 0.0001
learning_rate_con = 0.00001

alpha = 0.00003
total_error = 0 
total_correct = 0

num_epoch = 200
batch_size= 100

for iter in range(num_epoch):

    for current_batch_index in range(0,len(training_images),batch_size):

        current_batch = training_images[current_batch_index:current_batch_index+batch_size,:]
        current_batch_label = training_lables[current_batch_index:current_batch_index+batch_size,:]

        current_batch_reshape = np.reshape(current_batch,(batch_size,28,28))

        l1 = signal.convolve(current_batch_reshape, np.expand_dims(w1,axis=0)  , mode='valid')
        l1A = ReLu(l1)
        
        l2 = signal.convolve(l1A,np.expand_dims(w2,axis=0) , mode='valid')
        l2A = ReLu(l2)

        l3 = signal.convolve(l2A,np.expand_dims(w3,axis=0) , mode='valid')
        l3A = ReLu(l3)

        l4_Input = np.reshape(l3A,(batch_size,-1))
        l4 = l4_Input.dot(w4)
        l4A = arctan(l4)

        l5 = l4A.dot(w5)
        l5A = arctan(l5)

        l6 = l5A.dot(w6)
        l6A = log(l6)
        l6Soft = softmax(l6A)

        cost = ( -1 * (current_batch_label * np.log(l6Soft)  + ( 1- current_batch_label) * np.log(1 - l6Soft)  )).sum() / batch_size
        total_error+= cost

        grad_6_part_1 = l6Soft - current_batch_label
        grad_6_part_2 = d_log(l6)
        grad_6_part_3 = l5A
        grad_6 = grad_6_part_3.T.dot(grad_6_part_1*grad_6_part_2)
        
        grad_5_part_1 = (grad_6_part_1*grad_6_part_2 ).dot(w6.T)
        grad_5_part_2 = d_arctan(l5)
        grad_5_part_3 = l4A
        grad_5 = grad_5_part_3.T.dot(grad_5_part_1 * grad_5_part_2)
        
        grad_4_part_1 = (grad_5_part_1 * grad_5_part_2).dot(w5.T)
        grad_4_part_2 = d_arctan(l4)
        grad_4_part_3 = l4_Input
        grad_4 = grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

        grad_3_part_1 = np.reshape((grad_4_part_1 * grad_4_part_2).dot(w4.T),(batch_size,19,19))
        grad_3_part_2 = d_ReLu(l3)
        grad_3_part_3 = l2A
        grad_3 = np.rot90( signal.convolve(grad_3_part_3,   np.rot90(grad_3_part_1 * grad_3_part_2,2)    , mode='valid')    ,2)

        grad_2_part_1 = signal.convolve(np.expand_dims(w3,axis=0),   np.rot90(np.pad(grad_3_part_1 * grad_3_part_2,pad_width = ((0, 0), (2, 2), (2, 2)),mode='constant')  ,2),mode='valid' ) 
        grad_2_part_2 = d_ReLu(l2)
        grad_2_part_3 = l1A
        grad_2 = np.rot90( signal.convolve(grad_2_part_3,   np.rot90(grad_2_part_1 * grad_2_part_2,2)    , mode='valid')    ,2)

        grad_1_part_1 = signal.convolve(np.expand_dims(w2,axis=0),   np.rot90(np.pad(grad_2_part_1 * grad_2_part_2,pad_width = ((0, 0), (3, 3), (3, 3)),mode='constant')  ,2),mode='valid' ) 
        grad_1_part_2 = d_ReLu(l1)
        grad_1_part_3 = current_batch_reshape
        grad_1 = np.rot90( signal.convolve(grad_1_part_3,   np.rot90(grad_1_part_1 * grad_1_part_2,2)    , mode='valid')    ,2)

        w6 = w6- learning_rate  * grad_6
        w5 = w5- learning_rate* grad_5  
        w4 = w4- learning_rate* grad_4
        w3 = w3- learning_rate_con* np.squeeze(grad_3)
        w2 = w2- learning_rate_con* np.squeeze(grad_2)
        w1 = w1- learning_rate_con* np.squeeze(grad_1)


    if iter % 10 == 0 :
        print("current Iter: ", iter, " Current Cost :", total_error)

        for current_batch_index in range(10):

            testing_images,testing_lables = shuffle(testing_images,testing_lables)

            current_batch = testing_images[current_batch_index,:]
            current_batch_label = testing_lables[current_batch_index,:]

            current_batch_reshape = np.reshape(current_batch,(28,28))

            l1 = signal.convolve(current_batch_reshape, w1 , mode='valid')
            l1A = ReLu(l1)
            
            l2 = signal.convolve(l1A,w2, mode='valid')
            l2A = ReLu(l2)

            l3 = signal.convolve(l2A,w3, mode='valid')
            l3A = ReLu(l3)

            l4_Input = np.reshape(l3A,(-1))
            l4 = l4_Input.dot(w4)
            l4A = arctan(l4)

            l5 = l4A.dot(w5)
            l5A = arctan(l5)

            l6 = l5A.dot(w6)
            l6A = log(l6)
            l6Soft = softmax(l6A)

            print('Mid Training Test Current Predict : ',np.where(l6Soft == l6Soft.max()), " Ground Truth : ", np.where(current_batch_label == current_batch_label.max()) )
        print('------------------------')
    total_error = 0


for current_batch_index in range(len(testing_images)):

    current_batch = testing_images[current_batch_index,:]
    current_batch_label = testing_lables[current_batch_index,:]

    current_batch_reshape = np.reshape(current_batch,(28,28))

    l1 = signal.convolve(current_batch_reshape, w1 , mode='valid')
    l1A = ReLu(l1)
    
    l2 = signal.convolve(l1A,w2, mode='valid')
    l2A = ReLu(l2)

    l3 = signal.convolve(l2A,w3, mode='valid')
    l3A = ReLu(l3)

    l4_Input = np.reshape(l3A,(-1))
    l4 = l4_Input.dot(w4)
    l4A = arctan(l4)

    l5 = l4A.dot(w5)
    l5A = arctan(l5)

    l6 = l5A.dot(w6)
    l6A = log(l6)
    l6Soft = softmax(l6A)

    print(' Current Predict : ',np.where(l6Soft == l6Soft.max())[0].ravel(), " Ground Truth : ", np.where(current_batch_label == current_batch_label.max())[0].ravel() )
