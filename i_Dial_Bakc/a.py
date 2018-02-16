import numpy as np,sys,os
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
proportion_rate = 0.8
decay_rate = 0.1

num_epoch = 100
cost_array = []
total_cost = 0 
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]

# 2. Model to compare
class Diluted_RNN:
    def __init__(self,hidden_state,wx_h,wh_h):
        self.hidden_state = hidden_state.copy()
        self.wx_h = wx_h.copy()
        self.wh_h = wh_h.copy()

        hid_state_c = hidden_state.shape[0]
        hid_state_r = hidden_state.shape[1]
        wx_c = wx_h.shape[0]

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

    def backpropagation_with_gaussian(self,Time_Stamp,gradient,iter):
        limitation_time_stamp = Time_Stamp - 1
        grad_part_1 = gradient
        grad_part_2 = np.expand_dims(d_arctan(self.layers[limitation_time_stamp,:] ),axis=0)
        grad_part_x = np.expand_dims(self.input[limitation_time_stamp,:],axis=0)
        grad_part_h = np.expand_dims(self.hidden_state[limitation_time_stamp,:],axis=0)
        
        grad_x = grad_part_x.T.dot(grad_part_1 * grad_part_2)
        grad_h = grad_part_h.T.dot(grad_part_1 * grad_part_2)

        grad_pass_x = (grad_part_1 * grad_part_2).dot(self.wx_h.T)
        grad_pass_h = (grad_part_1 * grad_part_2).dot(self.wh_h.T)
        
        # ------ Calculate The Additive Noise -------
        n_value = 0.01
        ADDITIVE_NOISE_STD = n_value / (np.power((1 + iter), 0.55))
        ADDITIVE_GAUSSIAN_NOISE = np.random.normal(loc=0,scale=ADDITIVE_NOISE_STD)
        # ------ Calculate The Additive Noise -------

        self.wx_h = self.wx_h - LR_x_h * (grad_x + ADDITIVE_GAUSSIAN_NOISE)
        self.wh_h = self.wh_h - LR_h_h * (grad_h + ADDITIVE_GAUSSIAN_NOISE)
        return grad_pass_x,grad_pass_h    

# 3.1 Make the Common Weigths
hidden_state1 = np.zeros((6,128))
wx_h1 = np.random.randn(196,128)
wh_h1 = np.random.randn(128,128)

hidden_state2 = np.zeros((4,10))
wx_h2 = np.random.randn(128,10)
wh_h2 = np.random.randn(10,10)

# 4.1 Make each class  - Regular Back Propagation
layer1_case1 = Diluted_RNN(hidden_state1,wx_h1,wh_h1)
layer2_case1 = Diluted_RNN(hidden_state2,wx_h2,wh_h2)

# 4.2 Google Brain Guassian Noise
layer1_case2 = Diluted_RNN(hidden_state1,wx_h1,wh_h1)
layer2_case2 = Diluted_RNN(hidden_state2,wx_h2,wh_h2)

# 4.3 Dilated Back Propagation
layer1_case3 = Diluted_RNN(hidden_state1,wx_h1,wh_h1)
layer2_case3 = Diluted_RNN(hidden_state2,wx_h2,wh_h2)

# 4.4 Goole Brain, Dilated Back Propagation
layer1_case4 = Diluted_RNN(hidden_state1,wx_h1,wh_h1)
layer2_case4 = Diluted_RNN(hidden_state2,wx_h2,wh_h2)




# ======= CASE 1 =======
for iter in range(num_epoch):
    
    for image_index in range(len(training_images)):
        
        current_full_image = np.reshape(training_images[image_index,:],(28,28))
        current_label      = np.expand_dims(training_lables[image_index,:],axis=0)

        current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
        current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
        current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
        current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
        current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))
        
        l1_1 = layer1_case1.feed_forward(1,current_1_mean)
        _    = layer2_case1.feed_forward(1,l1_1)

        l2_1 = layer1_case1.feed_forward(2,current_2_block)

        l3_1 = layer1_case1.feed_forward(3,current_3_block)
        _    = layer2_case1.feed_forward(2,l3_1)

        l4_1 = layer1_case1.feed_forward(4,current_4_block)

        l5_1 = layer1_case1.feed_forward(5,current_5_block)
        output=layer2_case1.feed_forward(3,l5_1)

        output_Soft = softmax(output)
        cost = ( -1 * (current_label * np.log(output_Soft)  + ( 1- current_label) * np.log(1 - output_Soft)  )).sum() 
        print("Current Iter :",iter,"  Current Image Index:  ",image_index ," Real Time Update Cost: ", cost,end='\r')
        total_cost+= cost

        gradient_3_2_x,gradient_3_2_h = layer2_case1.backpropagation(3,output_Soft-current_label)
        _,             gradient_5_1_h = layer1_case1.backpropagation(5,gradient_3_2_x)

        _,             gradient_4_1_h = layer1_case1.backpropagation(4,gradient_5_1_h)

        gradient_2_2_x,gradient_2_2_h = layer2_case1.backpropagation(2,gradient_3_2_h)
        _,             gradient_3_1_h = layer1_case1.backpropagation(3,gradient_2_2_x+gradient_4_1_h)

        _,             gradient_2_1_h = layer1_case1.backpropagation(2,gradient_3_1_h)

        gradient_1_2_x,_              = layer2_case1.backpropagation(1,gradient_2_2_h)
        _,             _              = layer1_case1.backpropagation(1,gradient_1_2_x+gradient_2_1_h)
        
    if iter % 50 == 0 :
        print('\n======================================')
        print("current Iter: ", iter, " Current Total Cost :", total_cost/len(training_images))

        for current_batch_index in range(3):
    
            testing_images,testing_lables = shuffle(testing_images,testing_lables)
            current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
            current_label      = np.expand_dims(testing_lables[current_batch_index,:],axis=0)
            current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
            current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
            current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
            current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
            current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))

            l1_1 = layer1_case1.feed_forward(1,current_1_mean)
            _    = layer2_case1.feed_forward(1,l1_1)

            l2_1 = layer1_case1.feed_forward(2,current_2_block)

            l3_1 = layer1_case1.feed_forward(3,current_3_block)
            _    = layer2_case1.feed_forward(2,l3_1)

            l4_1 = layer1_case1.feed_forward(4,current_4_block)

            l5_1 = layer1_case1.feed_forward(5,current_5_block)
            output=layer2_case1.feed_forward(3,l5_1)
            output_Soft = softmax(output)

            print('Mid Training Test Current Predict : ',
            np.where(output_Soft == output_Soft.max())[0], 
            "  Ground Truth : ",
            np.where(current_label[0] == current_label[0].max())[0])
        print('======================================')

    cost_array.append(total_cost/len(training_images))
    total_cost = 0       
# ======= CASE 1 FINAL =======
print('==============FINAL CASE 1 ====================')
correct = 0
for current_batch_index in range(len(testing_images)):

    current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
    current_batch_label = np.expand_dims(testing_lables[current_batch_index,:],axis=0)
    current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
    current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
    current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
    current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
    current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))

    l1_1 = layer1_case1.feed_forward(1,current_1_mean)
    _    = layer2_case1.feed_forward(1,l1_1)

    l2_1 = layer1_case1.feed_forward(2,current_2_block)

    l3_1 = layer1_case1.feed_forward(3,current_3_block)
    _    = layer2_case1.feed_forward(2,l3_1)

    l4_1 = layer1_case1.feed_forward(4,current_4_block)

    l5_1 = layer1_case1.feed_forward(5,current_5_block)
    output=layer2_case1.feed_forward(3,l5_1)
    output_Soft = softmax(output)

    print(' Current Predict : ',np.where(output_Soft == output_Soft.max())[0], " Ground Truth : ", np.where(current_batch_label[0] == current_batch_label[0].max())[0] )
    if np.where(output_Soft == output_Soft.max())[0] == np.where(current_batch_label[0] == current_batch_label[0].max())[0]:
        correct += 1
case1_cost_array = cost_array
case1_correct = correct
cost_array = []
total_cost = 0 
print('=======================================\n')
# ======= CASE 1 FINAL =======




# ======= CASE 2 =======
for iter in range(num_epoch):
    
    for image_index in range(len(training_images)):
        
        current_full_image = np.reshape(training_images[image_index,:],(28,28))
        current_label      = np.expand_dims(training_lables[image_index,:],axis=0)

        current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
        current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
        current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
        current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
        current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))
        
        l1_1 = layer1_case2.feed_forward(1,current_1_mean)
        _    = layer2_case2.feed_forward(1,l1_1)

        l2_1 = layer1_case2.feed_forward(2,current_2_block)

        l3_1 = layer1_case2.feed_forward(3,current_3_block)
        _    = layer2_case2.feed_forward(2,l3_1)

        l4_1 = layer1_case2.feed_forward(4,current_4_block)

        l5_1 = layer1_case2.feed_forward(5,current_5_block)
        output=layer2_case2.feed_forward(3,l5_1)
        output_Soft = softmax(output)
        cost = ( -1 * (current_label * np.log(output_Soft)  + ( 1- current_label) * np.log(1 - output_Soft)  )).sum() 
        print("Current Iter :",iter,"  Current Image Index:  ",image_index ," Real Time Update Cost: ", cost,end='\r')
        total_cost+= cost

        gradient_3_2_x,gradient_3_2_h = layer2_case2.backpropagation_with_gaussian(3,output_Soft-current_label,iter)
        _,             gradient_5_1_h = layer1_case2.backpropagation_with_gaussian(5,gradient_3_2_x,iter)

        _,             gradient_4_1_h = layer1_case2.backpropagation_with_gaussian(4,gradient_5_1_h,iter)

        gradient_2_2_x,gradient_2_2_h = layer2_case2.backpropagation_with_gaussian(2,gradient_3_2_h,iter)
        _,             gradient_3_1_h = layer1_case2.backpropagation_with_gaussian(3,gradient_2_2_x+gradient_4_1_h,iter)

        _,             gradient_2_1_h = layer1_case2.backpropagation_with_gaussian(2,gradient_3_1_h,iter)

        gradient_1_2_x,_              = layer2_case2.backpropagation_with_gaussian(1,gradient_2_2_h,iter)
        _,             _              = layer1_case2.backpropagation_with_gaussian(1,gradient_1_2_x+gradient_2_1_h,iter)
        
    if iter % 50 == 0 :
        print('\n======================================')
        print("current Iter: ", iter, " Current Total Cost :", total_cost/len(training_images))

        for current_batch_index in range(3):
    
            testing_images,testing_lables = shuffle(testing_images,testing_lables)

            current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
            current_label      = np.expand_dims(testing_lables[current_batch_index,:],axis=0)
            current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
            current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
            current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
            current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
            current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))

            l1_1 = layer1_case2.feed_forward(1,current_1_mean)
            _    = layer2_case2.feed_forward(1,l1_1)

            l2_1 = layer1_case2.feed_forward(2,current_2_block)

            l3_1 = layer1_case2.feed_forward(3,current_3_block)
            _    = layer2_case2.feed_forward(2,l3_1)

            l4_1 = layer1_case2.feed_forward(4,current_4_block)

            l5_1 = layer1_case2.feed_forward(5,current_5_block)
            output=layer2_case2.feed_forward(3,l5_1)
            output_Soft = softmax(output)

            print('Mid Training Test Current Predict : ',
            np.where(output_Soft == output_Soft.max())[0], 
            "    Ground Truth : ",
            np.where(current_label[0] == current_label[0].max())[0] )
        print('======================================')

    cost_array.append(total_cost/len(training_images))
    total_cost = 0       
# ======= CASE 2 FINAL =======
print('==============FINAL CASE 2 ====================')
correct = 0
for current_batch_index in range(len(testing_images)):

    current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
    current_batch_label = np.expand_dims(testing_lables[current_batch_index,:],axis=0)
    current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
    current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
    current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
    current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
    current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))

    l1_1 = layer1_case2.feed_forward(1,current_1_mean)
    _    = layer2_case2.feed_forward(1,l1_1)

    l2_1 = layer1_case2.feed_forward(2,current_2_block)

    l3_1 = layer1_case2.feed_forward(3,current_3_block)
    _    = layer2_case2.feed_forward(2,l3_1)

    l4_1 = layer1_case2.feed_forward(4,current_4_block)

    l5_1 = layer1_case2.feed_forward(5,current_5_block)
    output=layer2_case2.feed_forward(3,l5_1)
    output_Soft = softmax(output)

    print(' Current Predict : ',np.where(output_Soft == output_Soft.max())[0], " Ground Truth : ", np.where(current_batch_label[0] == current_batch_label[0].max())[0] )
    if np.where(output_Soft == output_Soft.max())[0] == np.where(current_batch_label[0] == current_batch_label[0].max())[0]:
        correct += 1
case2_cost_array = cost_array
case2_correct = correct
cost_array = []
total_cost = 0 
print('=======================================\n')
# ======= CASE 2 FINAL =======




# ======= CASE 3 =======
for iter in range(num_epoch):
    
    for image_index in range(len(training_images)):
        
        current_full_image = np.reshape(training_images[image_index,:],(28,28))
        current_label      = np.expand_dims(training_lables[image_index,:],axis=0)
        current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
        current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
        current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
        current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
        current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))
        
        l1_1 = layer1_case3.feed_forward(1,current_1_mean)
        _    = layer2_case3.feed_forward(1,l1_1)

        l2_1 = layer1_case3.feed_forward(2,current_2_block)

        l3_1 = layer1_case3.feed_forward(3,current_3_block)
        _    = layer2_case3.feed_forward(2,l3_1)

        l4_1 = layer1_case3.feed_forward(4,current_4_block)

        l5_1 = layer1_case3.feed_forward(5,current_5_block)
        output=layer2_case3.feed_forward(3,l5_1)

        output_Soft = softmax(output)
        cost = ( -1 * (current_label * np.log(output_Soft)  + ( 1- current_label) * np.log(1 - output_Soft)  )).sum() 
        print("Current Iter :",iter,"  Current Image Index:  ",image_index ," Real Time Update Cost: ", cost,end='\r')
        total_cost+= cost

        decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter)

        gradient_3_2_x,gradient_3_2_h = layer2_case3.backpropagation(3,output_Soft-current_label)
        _,             gradient_5_1_h = layer1_case3.backpropagation(5,gradient_3_2_x)

        _,             gradient_4_1_h = layer1_case3.backpropagation(4,gradient_5_1_h)

        gradient_2_2_x,gradient_2_2_h = layer2_case3.backpropagation(2,gradient_3_2_h)
        _,             gradient_3_1_h = layer1_case3.backpropagation(3,gradient_2_2_x+gradient_4_1_h + decay_propotoin_rate * gradient_5_1_h)

        _,             gradient_2_1_h = layer1_case3.backpropagation(2,gradient_3_1_h + decay_propotoin_rate* gradient_4_1_h )

        gradient_1_2_x,_              = layer2_case3.backpropagation(1,gradient_2_2_h + decay_propotoin_rate * gradient_3_2_h)
        _,             _              = layer1_case3.backpropagation(1,gradient_1_2_x+gradient_2_1_h + decay_propotoin_rate * gradient_3_1_h)
        
    if iter % 50 == 0 :
        print('\n======================================')
        print("current Iter: ", iter, " Current Total Cost :", total_cost/len(training_images))

        for current_batch_index in range(3):
    
            testing_images,testing_lables = shuffle(testing_images,testing_lables)
            current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
            current_label      = np.expand_dims(testing_lables[current_batch_index,:],axis=0)
            current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
            current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
            current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
            current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
            current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))

            l1_1 = layer1_case3.feed_forward(1,current_1_mean)
            _    = layer2_case3.feed_forward(1,l1_1)

            l2_1 = layer1_case3.feed_forward(2,current_2_block)

            l3_1 = layer1_case3.feed_forward(3,current_3_block)
            _    = layer2_case3.feed_forward(2,l3_1)

            l4_1 = layer1_case3.feed_forward(4,current_4_block)

            l5_1 = layer1_case3.feed_forward(5,current_5_block)
            output=layer2_case3.feed_forward(3,l5_1)
            output_Soft = softmax(output)

            print('Mid Training Test Current Predict : ',
            np.where(output_Soft == output_Soft.max())[0], 
            "    Ground Truth : ",
            np.where(current_label[0] == current_label[0].max())[0])
        print('======================================')

    cost_array.append(total_cost/len(training_images))
    total_cost = 0       
# ======= CASE 3 FINAL =======
print('==============FINAL CASE 3 ====================')
correct = 0
for current_batch_index in range(len(testing_images)):

    current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
    current_batch_label = np.expand_dims(testing_lables[current_batch_index,:],axis=0)
    current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
    current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
    current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
    current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
    current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))

    l1_1 = layer1_case3.feed_forward(1,current_1_mean)
    _    = layer2_case3.feed_forward(1,l1_1)

    l2_1 = layer1_case3.feed_forward(2,current_2_block)

    l3_1 = layer1_case3.feed_forward(3,current_3_block)
    _    = layer2_case3.feed_forward(2,l3_1)

    l4_1 = layer1_case3.feed_forward(4,current_4_block)

    l5_1 = layer1_case3.feed_forward(5,current_5_block)
    output=layer2_case3.feed_forward(3,l5_1)
    output_Soft = softmax(output)

    print(' Current Predict : ',np.where(output_Soft == output_Soft.max())[0], " Ground Truth : ", np.where(current_batch_label[0] == current_batch_label[0].max())[0] )
    if np.where(output_Soft == output_Soft.max())[0] == np.where(current_batch_label[0] == current_batch_label[0].max())[0]:
        correct += 1
case3_cost_array = cost_array
case3_correct = correct
cost_array = []
total_cost = 0 
print('=======================================\n')
# ======= CASE 3 FINAL =======




# ======= CASE 4 =======
for iter in range(num_epoch):
    
    for image_index in range(len(training_images)):
        
        current_full_image = np.reshape(training_images[image_index,:],(28,28))
        current_label      = np.expand_dims(training_lables[image_index,:],axis=0)
        current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
        current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
        current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
        current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
        current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))
        
        l1_1 = layer1_case4.feed_forward(1,current_1_mean)
        _    = layer2_case4.feed_forward(1,l1_1)

        l2_1 = layer1_case4.feed_forward(2,current_2_block)

        l3_1 = layer1_case4.feed_forward(3,current_3_block)
        _    = layer2_case4.feed_forward(2,l3_1)

        l4_1 = layer1_case4.feed_forward(4,current_4_block)

        l5_1 = layer1_case4.feed_forward(5,current_5_block)
        output=layer2_case4.feed_forward(3,l5_1)

        output_Soft = softmax(output)
        cost = ( -1 * (current_label * np.log(output_Soft)  + ( 1- current_label) * np.log(1 - output_Soft)  )).sum() 
        print("Current Iter :",iter,"  Current Image Index:  ",image_index ," Real Time Update Cost: ", cost,end='\r')
        total_cost+= cost

        decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter)

        gradient_3_2_x,gradient_3_2_h = layer2_case4.backpropagation_with_gaussian(3,output_Soft-current_label,iter)
        _,             gradient_5_1_h = layer1_case4.backpropagation_with_gaussian(5,gradient_3_2_x,iter)

        _,             gradient_4_1_h = layer1_case4.backpropagation_with_gaussian(4,gradient_5_1_h,iter)

        gradient_2_2_x,gradient_2_2_h = layer2_case4.backpropagation_with_gaussian(2,gradient_3_2_h,iter)
        _,             gradient_3_1_h = layer1_case4.backpropagation_with_gaussian(3,gradient_2_2_x+gradient_4_1_h+ decay_propotoin_rate * gradient_5_1_h,iter)

        _,             gradient_2_1_h = layer1_case4.backpropagation_with_gaussian(2,gradient_3_1_h+ decay_propotoin_rate * gradient_4_1_h,iter)

        gradient_1_2_x,_              = layer2_case4.backpropagation_with_gaussian(1,gradient_2_2_h + decay_propotoin_rate * gradient_3_2_h,iter)
        _,             _              = layer1_case4.backpropagation_with_gaussian(1,gradient_1_2_x+gradient_2_1_h+decay_propotoin_rate * gradient_3_1_h,iter)
        
    if iter % 50 == 0 :
        print('\n======================================')
        print("current Iter: ", iter, " Current Total Cost :", total_cost/len(training_images))

        for current_batch_index in range(3):
    
            testing_images,testing_lables = shuffle(testing_images,testing_lables)

            current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
            current_label      = np.expand_dims(testing_lables[current_batch_index,:],axis=0)
            current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
            current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
            current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
            current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
            current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))

            l1_1 = layer1_case4.feed_forward(1,current_1_mean)
            _    = layer2_case4.feed_forward(1,l1_1)

            l2_1 = layer1_case4.feed_forward(2,current_2_block)

            l3_1 = layer1_case4.feed_forward(3,current_3_block)
            _    = layer2_case4.feed_forward(2,l3_1)

            l4_1 = layer1_case4.feed_forward(4,current_4_block)

            l5_1 = layer1_case4.feed_forward(5,current_5_block)
            output=layer2_case4.feed_forward(3,l5_1)
            output_Soft = softmax(output)

            print('Mid Training Test Current Predict : ',
            np.where(output_Soft == output_Soft.max())[0], 
            "    Ground Truth : ",
            np.where(current_label[0] == current_label[0].max())[0] )
        print('======================================')

    cost_array.append(total_cost/len(training_images))
    total_cost = 0       
# ======= CASE 4 FINAL =======
print('==============FINAL CASE 4 ====================')
correct = 0
for current_batch_index in range(len(testing_images)):

    current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
    current_batch_label = np.expand_dims(testing_lables[current_batch_index,:],axis=0)
    current_1_mean  = np.reshape(block_reduce(current_full_image,(2,2),np.mean),(1,-1))
    current_2_block = np.reshape(block_reduce(current_full_image,(2,2),np.var),(1,-1))
    current_3_block = np.reshape(block_reduce(current_full_image,(2,2),np.max),(1,-1))
    current_4_block = np.reshape(block_reduce(current_full_image,(2,2),np.std),(1,-1))
    current_5_block = np.reshape(block_reduce(current_full_image,(2,2),np.median),(1,-1))

    l1_1 = layer1_case4.feed_forward(1,current_1_mean)
    _    = layer2_case4.feed_forward(1,l1_1)

    l2_1 = layer1_case4.feed_forward(2,current_2_block)

    l3_1 = layer1_case4.feed_forward(3,current_3_block)
    _    = layer2_case4.feed_forward(2,l3_1)

    l4_1 = layer1_case4.feed_forward(4,current_4_block)

    l5_1 = layer1_case4.feed_forward(5,current_5_block)
    output=layer2_case4.feed_forward(3,l5_1)
    output_Soft = softmax(output)

    print(' Current Predict : ',np.where(output_Soft == output_Soft.max())[0], " Ground Truth : ", np.where(current_batch_label[0] == current_batch_label[0].max())[0] )
    if np.where(output_Soft == output_Soft.max())[0] == np.where(current_batch_label[0] == current_batch_label[0].max())[0]:
        correct += 1
case4_cost_array = cost_array
case4_correct = correct
cost_array = []
total_cost = 0 
print('=======================================\n')
# ======= CASE 4 FINAL =======










# Func: Display the Results
correct_array = [case1_correct,case2_correct,case3_correct,case4_correct]
plt.bar(range(len(correct_array)), correct_array,color=['r','g','b','m'])
plt.xticks(range(len(correct_array)), ('Case 1','Case 2','Case 3','Case 4'))
plt.title("Accuracy Bar Graph")
plt.legend()
plt.show()


plt.plot(range(len(case1_cost_array)), case1_cost_array, color='r',label='case 1')
plt.plot(range(len(case2_cost_array)), case2_cost_array, color='g',label='case 2')
plt.plot(range(len(case3_cost_array)), case3_cost_array, color='b',label='case 3')
plt.plot(range(len(case4_cost_array)), case4_cost_array, color='m',label='case 4')
plt.title("Cost over time Graph")
plt.legend()
plt.show()

# -- end code --

