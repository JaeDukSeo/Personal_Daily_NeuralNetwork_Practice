import csv,sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(5,suppress =True)
np.random.seed(3)

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

# 0. Read the csv file and replace all of the string
df = pd.read_csv('depression_preprocessed.csv',delimiter=',')
df.replace({'Imipramine': 0.3, 
'Lithium': 0.6, 
'Placebo': 0.9, 
'Recurrence': 0, 
'No Recurrence': 1}, inplace=True)

# 1. Read as Numpy Array - and shuffle it
full_data = df.as_matrix()
np.random.shuffle(full_data)


# 2. Split into Training and Test
traing_datas = full_data[:100,:4]
traing_label = full_data[:100,4:]
test_datas = full_data[100:,:4]
test_label = full_data[100:,4:]


hid_state_l1 = np.zeros((4,5))
wx_h_layer_1_c = np.random.randn(1,4) *0.07
wh_h_layer_1_c = np.random.randn(4,4) *0.07
wx_h_layer_1_g = np.random.randn(1,4) *0.07
wh_h_layer_1_g = np.random.randn(4,4) *0.07

hid_state_l2 = np.zeros((1,4))
wx_h_layer_2_c = np.random.randn(4,1) *0.1
wh_h_layer_2_c = np.random.randn(1,1) *0.1
wx_h_layer_2_g = np.random.randn(4,1) *0.1
wh_h_layer_2_g = np.random.randn(1,1) *0.1

# 3. Build a Modle with Shared Weights
class Dilated_Update_RNN():
    
    def __init__(self,hid_state=None,input_time=None,
        wx_h_c=None,wh_h_c=None,
        wx_h_g=None,wh_h_g=None):

        self.hid_state = hid_state.copy()
        self.wx_h_c,self.wh_h_c = wx_h_c.copy(),wh_h_c.copy()
        self.wx_h_g,self.wh_h_g = wx_h_g.copy(),wh_h_g.copy()

        self.input = np.zeros((self.wx_h_c.shape[0],input_time))

        self.layer_c = np.zeros((self.wx_h_c.shape[1],input_time))
        self.layer_cA = np.zeros((self.wx_h_c.shape[1],input_time))
        
        self.layer_g = np.zeros((self.wx_h_c.shape[1],input_time))
        self.layer_gA = np.zeros((self.wx_h_c.shape[1],input_time))
        
    def feed_forward(self,input=None,time_stamp = None):

        fake_time_for_matrix  = time_stamp - 1
        self.input[:,fake_time_for_matrix] = input
        
        self.layer_c[:,fake_time_for_matrix] = self.hid_state[:,fake_time_for_matrix].dot(self.wh_h_c) + \
                                               self.input[:,fake_time_for_matrix].dot(self.wx_h_c)
        self.layer_cA[:,fake_time_for_matrix] = arctan(self.layer_c[:,fake_time_for_matrix])

        self.layer_g[:,fake_time_for_matrix] = self.hid_state[:,fake_time_for_matrix].dot(self.wh_h_g) + \
                                               self.input[:,fake_time_for_matrix].dot(self.wx_h_g)
        self.layer_gA[:,fake_time_for_matrix] = tanh(self.layer_g[:,fake_time_for_matrix])

        self.hid_state[:,time_stamp] = self.layer_gA[:,fake_time_for_matrix] * self.hid_state[:,fake_time_for_matrix] + \
        (1.0 - self.layer_gA[:,fake_time_for_matrix]) * self.layer_cA[:,fake_time_for_matrix]
        return self.hid_state[:,time_stamp]

    def case1_back_prop(self,gradient=None,time_stamp = None):
        
        fake_time_for_matrix  = time_stamp - 1
        gradient_common  =      gradient

        # ===== Gradient Respect to G Gate =======
        grad_g_part_1 = self.hid_state[:,fake_time_for_matrix] - self.layer_cA[:,fake_time_for_matrix]
        grad_g_part_2 = d_tanh(self.layer_g[:,fake_time_for_matrix])
        grad_g_part_h = np.expand_dims(self.hid_state[:,fake_time_for_matrix],axis=0)
        grad_g_part_x = np.expand_dims(self.input[:,fake_time_for_matrix],axis=0)

        grad_g_h = grad_g_part_h.T.dot(np.expand_dims(gradient_common * grad_g_part_1 * grad_g_part_2 ,axis=0))
        grad_g_x = grad_g_part_x.T.dot(np.expand_dims(gradient_common * grad_g_part_1 * grad_g_part_2 ,axis=0))
        
        # ===== Gradient Respect to C Gate =======
        grad_c_part_1 = 1 - self.layer_gA[:,fake_time_for_matrix]
        grad_c_part_2 = d_arctan(self.layer_c[:,fake_time_for_matrix])
        grad_c_part_h = np.expand_dims(self.hid_state[:,fake_time_for_matrix],axis=0)
        grad_c_part_x = np.expand_dims(self.input[:,fake_time_for_matrix],axis=0)

        grad_c_h = grad_c_part_h.T.dot(np.expand_dims(gradient_common * grad_c_part_1 * grad_c_part_2 ,axis=0))
        grad_c_x = grad_c_part_x.T.dot(np.expand_dims(gradient_common * grad_c_part_1 * grad_c_part_2 ,axis=0))

        # ===== Gradient to Pass on ======
        grad_pass_h = self.layer_gA[:,fake_time_for_matrix]
        grad_pass_g_h = (grad_g_part_1 * grad_g_part_2).dot(self.wh_h_g.T)
        grad_pass_c_h = (grad_c_part_1 * grad_c_part_2).dot(self.wh_h_c.T)
        grad_pass_respect_to_h =  gradient_common + grad_pass_h + grad_pass_g_h + grad_pass_c_h

        grad_pass_g_h = (grad_g_part_1 * grad_g_part_2).dot(self.wx_h_g.T)
        grad_pass_c_h = (grad_c_part_1 * grad_c_part_2).dot(self.wx_h_c.T)
        grad_pass_respect_to_x = gradient_common + grad_pass_g_h + grad_pass_c_h

        # ==== Update the weights =====
        self.wx_h_c = self.wx_h_c - learning_rate_x * grad_c_x
        self.wh_h_c = self.wh_h_c - learning_rate_h * grad_c_h

        self.wx_h_g = self.wx_h_g - learning_rate_x * grad_g_x
        self.wh_h_g = self.wh_h_g - learning_rate_h * grad_g_h

        return grad_pass_respect_to_x,grad_pass_respect_to_h
        
        
layer1_case1 = Dilated_Update_RNN(hid_state_l1,4,
                                wx_h_layer_1_c,wh_h_layer_1_c,   
                                wx_h_layer_1_g,wh_h_layer_1_g)
layer2_case1 = Dilated_Update_RNN(hid_state_l2,3,
                                wx_h_layer_2_c,wh_h_layer_2_c,   
                                wx_h_layer_2_g,wh_h_layer_2_g)

# 4.Hyper Parameter
num_epoch = 300
learning_rate_h = 0.0006
learning_rate_x = 0.0006
total_cost = 0

cost_array_case_1 = []

for iter in range(num_epoch):

    for current_data_index in range(len(traing_datas)):
        
        current_data = traing_datas[current_data_index,:]
        current_label= traing_label[current_data_index,:]

        l1_1 = layer1_case1.feed_forward(current_data[0],1)
        _    = layer2_case1.feed_forward(l1_1,1)

        _    = layer1_case1.feed_forward(current_data[1],2)
        
        l1_3 = layer1_case1.feed_forward(current_data[2],3)
        _    = layer2_case1.feed_forward(l1_3,2)

        l1_4 = layer1_case1.feed_forward(current_data[3],4)
        out  = layer2_case1.feed_forward(l1_4,3)
        outLog = log(out)
        
        cost = np.square(outLog-current_label).sum() * 0.5
        print("Current Iter :",iter," Real Time Update Cost: ", cost,end='\r')
        total_cost += cost

        gradient_l2_3_x,gradient_l2_3_h = layer2_case1.case1_back_prop(d_log(out)*(outLog-current_label)   ,3)
        _,              gradient_l1_4_h = layer1_case1.case1_back_prop(gradient_l2_3_x,4)

        gradient_l2_2_x,gradient_l2_2_h = layer2_case1.case1_back_prop(gradient_l2_3_h,2)
        _,              gradient_l1_3_h = layer1_case1.case1_back_prop(gradient_l2_2_x+gradient_l1_4_h,3)

        _,              gradient_l1_2_h = layer1_case1.case1_back_prop(gradient_l1_3_h,2)

        gradient_l2_1_x,gradient_l2_1_h = layer2_case1.case1_back_prop(gradient_l2_2_h,1)
        _,              gradient_l1_1_h = layer1_case1.case1_back_prop(gradient_l2_1_x+gradient_l1_2_h,1)


    if iter % 50 == 0 :
        print('\n======================================')
        print("current Iter: ", iter, " Current Total Cost :", total_cost)
        for current_batch_index in range(len(test_datas)):
    
            current_data = test_datas[current_batch_index,:]
            current_label= test_label[current_batch_index,:]

            l1_1 = layer1_case1.feed_forward(current_data[0],1)
            _    = layer2_case1.feed_forward(l1_1,1)

            _    = layer1_case1.feed_forward(current_data[1],2)
            
            l1_3 = layer1_case1.feed_forward(current_data[2],3)
            _    = layer2_case1.feed_forward(l1_3,2)

            l1_4 = layer1_case1.feed_forward(current_data[3],4)
            out  = layer2_case1.feed_forward(l1_4,3)
            outLog = log(out)

            print("Predicted: ",outLog, " Predict Rounded: " , np.round(outLog) ," GT : ",current_label )

        print('======================================')

    cost_array_case_1.append(total_cost)
    total_cost = 0

print('==============FINAL Prediction====================')
correct = 0
for current_batch_index in range(len(test_datas)):

    current_data = test_datas[current_batch_index,:]
    current_label= test_label[current_batch_index,:]

    l1_1 = layer1_case1.feed_forward(current_data[0],1)
    _    = layer2_case1.feed_forward(l1_1,1)

    _    = layer1_case1.feed_forward(current_data[1],2)
    
    l1_3 = layer1_case1.feed_forward(current_data[2],3)
    _    = layer2_case1.feed_forward(l1_3,2)

    l1_4 = layer1_case1.feed_forward(current_data[3],4)
    out  = layer2_case1.feed_forward(l1_4,3)
    outLog = log(out)

    print("Predicted: ",outLog, " Predict Rounded: " , np.round(outLog) ," GT : ",current_label )

    if np.round(outLog) == current_label:
        correct = correct + 1

print('Correct : ',correct, ' Out of : ',len(test_datas) )
plt.title("Cost over time")
plt.plot(np.arange(len(cost_array_case_1)), cost_array_case_1)
plt.show()

# -- end code --