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
        print("Current Iter :",iter,"  Current Image Index:  ",image_index ," Real Time Update Cost: ", cost,end='\r')
        total_cost+= cost

        gradient_3_2_x,gradient_3_2_h = layer2.backpropagation(3,output_Soft-current_label)
        _,             gradient_5_1_h = layer1.backpropagation(5,gradient_3_2_x)

        _,             gradient_4_1_h = layer1.backpropagation(4,gradient_5_1_h)

        gradient_2_2_x,gradient_2_2_h = layer2.backpropagation(2,gradient_3_2_h)
        _,             gradient_3_1_h = layer1.backpropagation(3,gradient_2_2_x+gradient_4_1_h)

        _,             gradient_2_1_h = layer1.backpropagation(2,gradient_3_1_h)

        gradient_1_2_x,_              = layer2.backpropagation(1,gradient_2_2_h)
        _,             _              = layer1.backpropagation(1,gradient_1_2_x+gradient_2_1_h)
        
    if iter % 2 == 0 :
        print('\n======================================')
        print("current Iter: ", iter, " Current Total Cost :", total_cost/len(training_images))

        for current_batch_index in range(5):
    
            testing_images,testing_lables = shuffle(testing_images,testing_lables)

            current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
            current_label      = np.expand_dims(testing_lables[current_batch_index,:],axis=0)

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


            print('Mid Training Test Current Predict : ',
            np.where(output_Soft == output_Soft.max())[0], 
            "    Ground Truth : ",
            np.where(current_label[0] == current_label[0].max())[0] ,'\n',
            "Array : ",output_Soft,'\n'
            "Array: ",current_label)
        print('======================================')

    cost_array.append(total_cost/len(training_images))
    total_cost = 0       


print('=======================================')
print('==============FINAL====================')
correct = 0
for current_batch_index in range(len(testing_images)):

    current_full_image = np.reshape(testing_images[current_batch_index,:],(28,28))
    current_batch_label = np.expand_dims(testing_lables[current_batch_index,:],axis=0)

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

    print(' Current Predict : ',np.where(output_Soft == output_Soft.max())[0], 
    " Ground Truth : ", np.where(current_batch_label[0] == current_batch_label[0].max())[0] )

    if np.where(output_Soft == output_Soft.max())[0] == np.where(current_batch_label[0] == current_batch_label[0].max())[0]:
        correct += 1

print('Correct : ',correct, ' Out of : ',len(testing_images) )
plt.title("Cost over time")
plt.plot(np.arange(len(cost_array)), cost_array)
plt.show()

# -- end code --