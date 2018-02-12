import numpy as np,sys
import tensorflow as tf
from sklearn.utils import shuffle
from scipy.signal import convolve2d
import skimage.measure
import matplotlib.pyplot as plt
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
test_image_num,training_image_num = 50,330
learning_rate = 0.02
num_epoch = 1
cost_array = []
batch_size = 3
total_cost = 0 
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]


class HighwayLayer:
    
    def __init__(self,layernum,dim1,dim2):
        
        self.wH = np.random.randn(dim1,dim2) * 0.5
        self.wT = np.random.randn(dim1,dim2) * 0.5
        self.wA = np.random.randn(dim1,dim2) * 0.5
        # self.wA = np.ones((dim1,dim2))

        self.input  = None
        self.output = None
        
        self.layer_H,self.layer_HA = None,None
        self.layer_T,self.layer_TA = None,None
        self.layer_A,self.layer_AA = None,None

    def feed_forward(self,current_input=None):
        
        self.input = current_input

        self.layer_H  = self.input.dot(self.wH) 
        self.layer_HA = arctan(self.layer_H)

        self.layer_T  = self.input.dot(self.wT) 
        self.layer_TA = log(self.layer_T)

        self.layer_A  = self.input.dot(self.wA) 
        self.layer_AA = tanh(self.layer_A)

        self.output = self.layer_HA * self.layer_TA + self.layer_AA * (1 - self.layer_TA)
        return  self.output

    def back_propagation(self,gradient=None):
        
        grad_part_common = gradient

        grad_part_2_A = (1 - self.layer_TA) * d_tanh(self.layer_A)
        grad_part_A = self.input.T.dot(grad_part_common * grad_part_2_A)

        grad_part_2_T = (self.layer_HA  - self.layer_AA) * d_log(self.layer_T)
        grad_part_T = self.input.T.dot(grad_part_common * grad_part_2_T)

        grad_part_2_H = self.layer_TA * d_arctan(self.layer_H)
        grad_part_H = self.input.T.dot(grad_part_common * grad_part_2_H)

        grad_pass_on =   (grad_part_common * grad_part_2_A).dot(self.wA.T) + \
                         (grad_part_common * grad_part_2_T).dot(self.wT.T) + \
                         (grad_part_common * grad_part_2_H).dot(self.wH.T) 

        self.wA = self.wA - learning_rate * grad_part_A
        self.wT = self.wT - learning_rate * grad_part_T
        self.wH = self.wH - learning_rate * grad_part_H
        
        return grad_pass_on

layer1 = HighwayLayer(1,784,1024)
layer2 = HighwayLayer(2,1024,824)
layer3 = HighwayLayer(3,824,256)
layer4 = HighwayLayer(3,256,10)

for iter in range(num_epoch):
    
    for currnet_image_index in range(0,len(training_images),batch_size):

        current_image = training_images[currnet_image_index:currnet_image_index+batch_size,:]
        current_label = training_lables[currnet_image_index:currnet_image_index+batch_size,:]

        l1 = layer1.feed_forward(current_image)
        l2 = layer2.feed_forward(l1)
        l3 = layer3.feed_forward(l2)
        l4 = layer4.feed_forward(l3)
        l4Soft = softmax(l4)

        cost = ( -1 * (current_label * np.log(l4Soft) + ( 1-current_label ) * np.log(1 - l4Soft)    )).sum() 
        print("Real Time Update Cost: ", cost,end='\r')
        total_cost += cost

        grad_4 = layer4.back_propagation(l4Soft - current_label)
        grad_3 = layer3.back_propagation(grad_4)
        grad_2 = layer2.back_propagation(grad_3)
        _ = layer1.back_propagation(grad_2)

    if iter % 2 == 0 :
        print("current Iter: ", iter, " Current Cost :", total_cost/len(training_images))

        for current_batch_index in range(0,6,batch_size):
    
            testing_images,testing_lables = shuffle(testing_images,testing_lables)

            current_batch =       testing_images[current_batch_index:current_batch_index+batch_size,:]
            current_batch_label = testing_lables[current_batch_index:current_batch_index+batch_size,:]

            l1 = layer1.feed_forward(current_batch)
            l2 = layer2.feed_forward(l1)
            l3 = layer3.feed_forward(l2)
            l4 = layer4.feed_forward(l3)
            l4Soft = softmax(l4)

            print(
            'Current Predict : ',np.where(l4Soft[0,:] == l4Soft[0,:].max())[0], 
            "Ground Truth   : ",np.where(current_batch_label[0,:] == current_batch_label[0,:].max())[0],'\n',
            'Current Predict : ',np.where(l4Soft[1,:] == l4Soft[1,:].max())[0], 
            "Ground Truth   : ",np.where(current_batch_label[1,:] == current_batch_label[1,:].max())[0],'\n',
            'Current Predict : ',np.where(l4Soft[2,:] == l4Soft[2,:].max())[0], 
            "Ground Truth   : ",np.where(current_batch_label[2,:] == current_batch_label[2,:].max())[0]
            )
        print('---------------------------------')
    cost_array.append(total_cost/len(training_images))
    total_cost = 0


print('=======================================')
print('==============FINAL====================')
correct = 0
for current_batch_index in range(len(testing_images)):

    current_batch =       np.expand_dims(testing_images[current_batch_index,:],axis=0)
    current_batch_label = np.expand_dims(testing_lables[current_batch_index,:],axis=0)

    l1 = layer1.feed_forward(current_batch)
    l2 = layer2.feed_forward(l1)
    l3 = layer3.feed_forward(l2)
    l4 = layer4.feed_forward(l3)
    l4Soft = softmax(l4)

    print(' Current Predict : ',np.where(l4Soft[0] == l4Soft[0].max())[0], 
    " Ground Truth : ", np.where(current_batch_label == current_batch_label.max())[0] )

    if np.where(l4Soft[0] == l4Soft[0].max())[0] == np.where(current_batch_label == current_batch_label.max())[0]:
        correct += 1

print('Correct : ',correct, ' Out of : ',len(testing_images) )
plt.title("Cost over time")
plt.plot(np.arange(len(cost_array)), cost_array)
plt.show()

# -- end code --