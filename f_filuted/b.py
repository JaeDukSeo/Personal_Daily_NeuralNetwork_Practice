import numpy as np,dicom,sys,os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage.measure import block_reduce
from tensorflow.examples.tutorials.mnist import input_data

np.random.randn(6789)

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
test_image_num,training_image_num = 50,300

# 2. Declare Hyper Parameters
LR_x_h1  = 0.001
LR_h1_h1 = 0.0001

LR_h1_h2 = 0.00001
LR_h2_h2 = 0.000001

num_epoch = 1
cost_array = []
batch_size = 3
total_cost = 0 
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]

hidden_layer_1 = np.zeros((6,86))
hidden_layer_2 = np.zeros((4,10))

wx_h1  = np.random.randn(196,86)
wh1_h1 = np.random.randn(86,86)

wh1_h2 = np.random.rand(86,10)
wh2_h2 = np.random.rand(86,10)


# 3. Build Model
class Diluted_RNN:
    
    def __init__(self,hid_state_c,hid_state_r,wx_c,wx_r,wh_c,wh_r):
        self.hidden_state = np.zeros((hid_state_c,hid_state_r))
        self.wx_h = np.random.randn(wx_c,wx_r)
        self.wh_h = np.random.randn(wh_c,wh_r)

    def feed_forward(self,Time_Stamp,input):
        
        layer  = input.dot(self.wx_h) + self.hidden_state[Time_Stamp,:].dot(self.wh_h)
        layerA = arctan(layer)

        return layerA

    def backpropagation(self,Time_Stamp,gradient):
        print("bakc")

# 4. Make the object
layer1 = Diluted_RNN(6,86,196,86,86,86)
layer2 = Diluted_RNN(4,10,86, 10,10,10)


# Func: Train all of the models
for iter in range(num_epoch):
    
    for image_index in range(len(training_images)):
        
        current_full_image = np.reshape(training_images[image_index,:],(28,28))
        current_label      = np.expand_dims(training_lables[image_index,:],axis=0)

        current_1_mean    = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
        current_2_block = np.reshape(current_full_image[:14,:14],(1,-1))
        current_3_block = np.reshape(current_full_image[:14,14:],(1,-1))
        current_4_block = np.reshape(current_full_image[14:,:14],(1,-1))
        current_5_block = np.reshape(current_full_image[14:,14:],(1,-1))
        
        l1_1 = layer1.feed_forward(1,current_1_mean)
        _    = layer2.feed_forward(1,l1_1)

        l2_1 = layer1.feed_forward(2,current_2_block)

        l3_1 = layer1.feed_forward(3,current_3_block)
        _    = layer2.feed_forward(2,l3_1)

        l4_1 = layer1.feed_forward(4,current_4_block)

        l5_1 = layer1.feed_forward(5,current_5_block)
        output=layer2.feed_forward(3,l5_1)

        output_Soft = softmax(output)
        cost = ( -1 * (current_label * np.log(output_Soft)  + ( 1- current_label) * np.log(1 - output_Soft)  )).sum() 
        print(cost)
        
        

        sys.exit()


# -- end code --