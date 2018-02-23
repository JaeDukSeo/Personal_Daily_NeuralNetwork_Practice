import numpy as np,dicom,sys,os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage.measure import block_reduce
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd

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
# mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True).test
# images,label = shuffle(mnist.images,mnist.labels)
# test_image_num,training_image_num = 100,3000

# 1.5 Read the Real Data
df = pd.read_csv('depression_processed.csv')
df.replace({'Imipramine': 1,'Lithium':2,'Placebo':3,'Recurrence':0,'No Recurrence':1}, inplace=True)

depression_full_data = shuffle(df.as_matrix())

depression_test  = depression_full_data[:9,:]
depression_train = depression_full_data[9:,:]

depression_test_label    = np.expand_dims(depression_test[:,-1],axis=1)
depression_test_features = depression_test[:,:4]

depression_train_label    = np.expand_dims(depression_train[:,-1],axis=1)
depression_train_features = depression_train[:,:4]


# 2. Declare Hyper Parameters
LR_x_h  = 0.001
LR_h_h =  0.001

num_epoch = 1
cost_array = []
total_cost = 0 

# 3. Build Model
class Diluted_Update_Gate_RNN:
    
    def __init__(self,hid_state_c,hid_state_r,  wx_c,wx_r,  wh_c,wh_r):
        self.hidden_state = np.zeros((hid_state_c,hid_state_r))

        self.wx_hc = np.random.randn(wx_c,wx_r)
        self.wh_hc = np.random.randn(wh_c,wh_r)
        self.wx_hg = np.random.randn(wx_c,wx_r)
        self.wh_hg = np.random.randn(wh_c,wh_r)

        self.input  =  np.zeros((wx_c,hid_state_r))

        self.layerc =  np.zeros((hid_state_c,hid_state_r-1))
        self.layercA=  np.zeros((hid_state_c,hid_state_r-1))
        self.layerg =  np.zeros((hid_state_c,hid_state_r-1))
        self.layergA=  np.zeros((hid_state_c,hid_state_r-1))

    def feed_forward(self,Time_Stamp,input):
        
        limitation_time_stamp = Time_Stamp - 1
        self.input[:,limitation_time_stamp]  = input

        self.layerc[:,limitation_time_stamp]  = self.hidden_state[:,limitation_time_stamp].dot(self.wh_hc) + \
                                                self.input[:,limitation_time_stamp].dot(self.wx_hc)
        self.layercA[:,limitation_time_stamp] = log(self.layerc[:,limitation_time_stamp])
        
        self.layerg[:,limitation_time_stamp]  = self.hidden_state[:,limitation_time_stamp].dot(self.wh_hg) + \
                                                self.input[:,limitation_time_stamp].dot(self.wx_hg)
        self.layergA[:,limitation_time_stamp] = arctan(self.layerc[:,limitation_time_stamp])

        self.hidden_state[:,Time_Stamp] = self.layergA[:,limitation_time_stamp] * self.hidden_state[:,limitation_time_stamp] + \
                                         ( 1-self.layergA[:,limitation_time_stamp] ) * self.layercA[:,limitation_time_stamp] 

        return self.hidden_state[:,Time_Stamp]

    def backpropagation(self,Time_Stamp,gradient):
        
        limitation_time_stamp = Time_Stamp - 1
        grad_part_1 = gradient

        grad_c_part_1 = ( 1-self.layergA[:,limitation_time_stamp] )
        grad_c_part_2 = d_log(self.layerc[:,limitation_time_stamp])
        grad_c_part_h = self.hidden_state[:,limitation_time_stamp]
        grad_c_part_x = self.input[:,limitation_time_stamp]       

        grad_c_h = grad_c_part_h.T.dot(gradient * grad_c_part_1 * grad_c_part_2)
        grad_c_x = grad_c_part_x.T.dot(gradient * grad_c_part_1 * grad_c_part_2)


        sys.exit()        



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
layer1 = Diluted_Update_Gate_RNN(4,5,  1,4,  4,4)
layer2 = Diluted_Update_Gate_RNN(1,4,  4,1,  1,1)

# Func: Train all of the models
for iter in range(num_epoch):
    
    for depression_index in range(len(depression_train_features)):
        
        current_depression_data = np.expand_dims(depression_train_features[depression_index,:],axis=1).T
        current_depression_label= depression_train_label[depression_index,:]

        l1_1 = np.expand_dims(layer1.feed_forward(1,current_depression_data[0,0]),axis=0)
        _    = layer2.feed_forward(1,l1_1)
        
        l1_2 = np.expand_dims(layer1.feed_forward(2,current_depression_data[0,1]),axis=0)

        l1_3 = np.expand_dims(layer1.feed_forward(3,current_depression_data[0,2]),axis=0)
        _    = layer2.feed_forward(2,l1_3)

        l1_4 =   np.expand_dims(layer1.feed_forward(4,current_depression_data[0,3]),axis=0)
        output = layer2.feed_forward(3,l1_4)
        otuput_log = log(output)

        cost = np.square(otuput_log -current_depression_label).sum() * 0.5
        print("Read Time Cost Iter: ", iter, " cost : ",cost,end='\r')
        total_cost+= cost

        layer2.backpropagation(3,(otuput_log-current_depression_label) * d_log(output))
        
        
        


        sys.exit()
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