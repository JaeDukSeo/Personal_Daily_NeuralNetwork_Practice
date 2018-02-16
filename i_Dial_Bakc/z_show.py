import numpy as np,dicom,sys,os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage.measure import block_reduce
from tensorflow.examples.tutorials.mnist import input_data

np.random.randn(6789)
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
test_image_num,training_image_num = 100,3000

# 2. Declare Hyper Parameters
LR_x_h  = 0.001
LR_h_h =  0.001

num_epoch = 100
cost_array = []
total_cost = 0 
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]

# 3. Build Model
class Diluted_RNN:
    
    def __init__(self,hid_state_c,hid_state_r,  wx_c,wx_r,  wh_c,wh_r):
        self.hidden_state = np.zeros((hid_state_c,hid_state_r))
        self.wx_h = np.random.randn(wx_c,wx_r)
        self.wh_h = np.random.randn(wh_c,wh_r)

        self.input  =  np.zeros((hid_state_c-1,wx_c))
        self.layers =  np.zeros((hid_state_c-1,hid_state_r))
        self.layersA=  np.zeros((hid_state_c-1,hid_state_r))

    def feed_forward(self,Time_Stamp,input):
        limitation_time_stamp = Time_Stamp - 1
        self.input[limitation_time_stamp,:]   = input
        self.layers[limitation_time_stamp,:]  = self.input[limitation_time_stamp,:].dot(self.wx_h) + self.hidden_state[limitation_time_stamp,:].dot(self.wh_h)
        self.layersA[limitation_time_stamp,:] = arctan(self.layers[limitation_time_stamp,:])
        return self.layersA[limitation_time_stamp,:]

    def backpropagation(self,Time_Stamp,gradient):
        
        limitation_time_stamp = Time_Stamp - 1
        grad_part_1 = gradient
        grad_part_2 = np.expand_dims(d_arctan(self.layers[limitation_time_stamp,:] ),axis=0)
        grad_part_x = np.expand_dims(self.input[limitation_time_stamp,:],axis=0)
        grad_part_h = np.expand_dims(self.hidden_state[limitation_time_stamp,:],axis=0)
        
        grad_x = grad_part_x.T.dot(grad_part_1 * grad_part_2)
        grad_h = grad_part_h.T.dot(grad_part_1 * grad_part_2)

        grad_pass_x = (grad_part_1 * grad_part_2).dot(self.wx_h.T)
        grad_pass_h = (grad_part_1 * grad_part_2).dot(self.wh_h.T)
        
        self.wx_h = self.wx_h - LR_x_h * grad_x 
        self.wh_h = self.wh_h - LR_h_h * grad_h 

        return grad_pass_x,grad_pass_h

# 4. Make the object
layer1 = Diluted_RNN(6,128,  196,128,  128,128)
layer2 = Diluted_RNN(4,10,  128, 10,  10,10)

# Func: Train all of the models
for iter in range(num_epoch):
    
    for image_index in range(len(training_images)):
        
        current_full_image = np.reshape(training_images[image_index,:],(28,28))
        current_label      = np.expand_dims(training_lables[image_index,:],axis=0)

        current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
        current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
        current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
        current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
        current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))

        plt.imshow(np.reshape(current_1_mean,(14,14)),cmap='gray')
        plt.title('Mean')
        plt.show()

        plt.imshow(np.reshape(current_2_block,(14,14)),cmap='gray')
        plt.title('var')
        plt.show()

        plt.imshow(np.reshape(current_3_block,(14,14)),cmap='gray')
        plt.title('max')
        plt.show()

        plt.imshow(np.reshape(current_4_block,(14,14)),cmap='gray')
        plt.title('STD')
        plt.show()

        plt.imshow(np.reshape(current_5_block,(14,14)),cmap='gray')
        plt.title('Mdian')
        plt.show()
                


        sys.exit()