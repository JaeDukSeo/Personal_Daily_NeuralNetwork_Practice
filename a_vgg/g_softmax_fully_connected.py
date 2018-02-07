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
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()

def softmax2(x):
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


test_image_num,training_image_num = 10,100
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]
num_epoch = 100

learning_rate = 0.01
total_cost = 0
w1 = np.random.randn(784,824) * 0.02
w2 = np.random.randn(824,1024)* 0.02
w3 = np.random.randn(1024,1240)* 0.02
w4 = np.random.randn(1240,5)* 0.02


for iter in range(num_epoch):
    
    for batch_size in range(len(training_images)):
        
        current_batch_image  = np.expand_dims(training_images[batch_size,:],axis=0)
        current_batch_label  = np.expand_dims(training_lables[batch_size,:],axis=0)

        l1 = current_batch_image.dot(w1)
        l1A = arctan(l1)

        l2 = l1A.dot(w2)
        l2A = arctan(l2)

        l3 = l2A.dot(w3)
        l3A = arctan(l3)

        l4 = l3A.dot(w4)
        l4Soft = softmax(l4)
        # l4Soft = softmax2(l4)

        cost = ( -1 * (current_batch_label * np.log(l4Soft)  + ( 1- current_batch_label) * np.log(1 - l4Soft)  )).sum() 
        total_cost += cost

        grad_4_part_1 = l4Soft - current_batch_label
        grad_4_part_2 = d_arctan(l4)
        grad_4_part_3 = l3A
        grad_4 =         grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

        grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
        grad_3_part_2 = d_arctan(l3)
        grad_3_part_3 = l2A
        grad_3 =         grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)
        
        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_arctan(l2)
        grad_2_part_3 = l1A
        grad_2 =         grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
        
        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_arctan(l1)
        grad_1_part_3 = current_batch_image
        grad_1 =         grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w4 = w4 - learning_rate * grad_4
        w3 = w3 - learning_rate * grad_3
        w2 = w2 - learning_rate * grad_2
        w1 = w1 - learning_rate * grad_1
        
    if iter % 10 == 0 :
        print("current Iter: ", iter, " Current Cost :", total_cost)

        for current_batch_index in range(10):

            testing_images,testing_lables = shuffle(testing_images,testing_lables)

            current_batch = np.expand_dims(testing_images[current_batch_index,:],axis=0)
            current_batch_label = testing_lables[current_batch_index,:]

            l1 = current_batch.dot(w1)
            l1A = arctan(l1)

            l2 = l1A.dot(w2)
            l2A = arctan(l2)

            l3 = l2A.dot(w3)
            l3A = arctan(l3)

            l4 = l3A.dot(w4)
            l4Soft = softmax(l4)
            # l4Soft = softmax2(l4)

            print('Mid Training Test Current Predict : ',
            np.where(l4Soft[0] == l4Soft[0].max())[0], '\n',
            "Array : ",l4Soft,
            " Ground Truth : ",
            np.where(current_batch_label == current_batch_label.max())[0] ,'\n',
            "array: ",current_batch_label)
        print('------------------------')
    total_cost = 0



# -- end code --