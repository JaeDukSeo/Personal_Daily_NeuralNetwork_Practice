import numpy as np,sys
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(5678)
np.set_printoptions(precision=3,suppress=True)


def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))

def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()
def d_softmax(x):
    return 0

# 0. Declare Training Data and Labels
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

train = mnist.test
images, labels = train.images, train.labels
images,label = shuffle(images,labels)

num_epoch = 200
total_cost = 0
alpha = 0.09
test_image_num,training_image_num = 20,300
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]

w1 = np.random.randn(784,256)
w2 = np.random.randn(256,128)
w3 = np.random.randn(128,10)

v1,v2,v3 = 0,0,0
m1,m2,m3 = 0,0,0

lr= 0.001
beta1 = 0.9
beta2 = 0.999
adam_e = 0.00000001
for iter in range(num_epoch):
    
    for current_image_index in range(len(training_images)):
        
        current_image = np.expand_dims(training_images[current_image_index],axis=0)
        current_image_label =  np.expand_dims(training_lables[current_image_index],axis=0)

        l1 = current_image.dot(w1)
        l1A = tanh(l1)

        l2 = l1A.dot(w2)
        l2A = tanh(l2)

        l3 = l2A.dot(w3)
        l3A = tanh(l3)

        l3Soft = softmax(l3A)

        cost =  (-1 * ( current_image_label * np.log(l3Soft) - ( 1- current_image_label) * log(1- l3Soft)  )).sum()
        # print("Current Iter: ", iter, " Current cost: ", cost)
        total_cost = total_cost + cost

        grad_3_part_1 = l3Soft - current_image_label
        grad_3_part_2 = d_tanh(l3)
        grad_3_part_3 = l2A
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_tanh(l2)
        grad_2_part_3 = l1A
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_tanh(l1)
        grad_1_part_3 = current_image
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        m1 = beta1*m1+ ( 1- beta1) * grad_1
        m2 = beta1*m2 + ( 1- beta1)* grad_2
        m3 = beta1 *m3+ ( 1- beta1)* grad_3
        
        v1 = beta2*v1+ ( 1- beta2) * grad_1**2
        v2 = beta2*v2 + ( 1- beta2)* grad_2**2
        v3 = beta2*v3+ ( 1- beta2)* grad_3 **2
        
        m1_hat = m1/ ( 1- beta1)
        m2_hat = m2/( 1- beta1)
        m3_hat = m3/( 1- beta1)
        
        v1_hat = v1/(1-beta2)
        v2_hat = v2/(1-beta2)
        v3_hat = v3/(1-beta2)

        w1 = w1 - (lr/( np.sqrt(v1_hat) + adam_e ) ) * m1_hat
        w2 = w2 - (lr/( np.sqrt(v2_hat) + adam_e ) ) * m2_hat
        w3 = w3 - (lr/( np.sqrt(v3_hat) + adam_e ) ) * m3_hat
        

    if iter % 10 == 0 : 
        print("Current Iter: ", iter, " Current cost: ", total_cost)
    total_cost = 0 




for current_image_index in range(len(testing_images)):

    current_image = np.expand_dims(testing_images[current_image_index],axis=0)
    current_image_label =  np.expand_dims(testing_lables[current_image_index],axis=0)

    l1 = current_image.dot(w1)
    l1A = tanh(l1)

    l2 = l1A.dot(w2)
    l2A = tanh(l2)

    l3 = l2A.dot(w3)
    l3A = tanh(l3)

    l3Soft = softmax(l3A)

    print("Predict : ",l3Soft)
    print("Predict round : ",np.round(l3Soft))
    print("GT  : ",current_image_label )
    print('--------------------')
    
    














# --- end code ---
