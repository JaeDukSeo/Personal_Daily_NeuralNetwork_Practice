import tensorflow as tf
import numpy as np,sys
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os
from read_10_data import get_data

np.random.seed(6789)
tf.set_random_seed(678)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def tf_Relu(x): return tf.nn.elu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_acrtan(x): return tf.atan(x)
def d_tf_arctan(x): return 1/(1+tf.square(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_elu(x): return tf.nn.elu(x)

def tf_softmax(x): return tf.nn.softmax(x)

# make class
class CNNLayer():
    
    def __init__(self,kernel,in_c,out_c,act,d_act):
        with tf.device('/cpu:0'):
            self.w = tf.Variable(tf.random_normal([kernel,kernel,in_c,out_c]))
            self.act,self.d_act = act,d_act
            self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))
    def getw(self): return [self.w]

    def feedforward(self,input,padding):
        self.input  = input 
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding=padding)
        self.layerA = self.act(self.layer)
        return self.layerA

    def feedforward_mean(self,input,padding):
        self.input  = input 
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding=padding)
        self.layerA = self.act(self.layer)
        self.layerA_mean = tf.nn.avg_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        # self.layerA_mean = tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        return self.layerA_mean

    def backprop(self,gradient,padding):
        
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,
            out_backprop = grad_middle,
            strides = [1,1,1,1],
            padding=padding
        )

        pass_on_grad = tf.nn.conv2d_backprop_input(
            input_sizes= [batch_size, self.input.shape[1].value, self.input.shape[2].value, self.input.shape[3].value],
            filter = self.w,
            out_backprop = grad_middle,
            strides = [1,1,1,1],
            padding=padding
        )

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return pass_on_grad,grad_update             

class FNNLayer():
    
    def __init__(self,in_c,out_c,act,d_act):
        with tf.device('/cpu:0'):
            self.w = tf.Variable(tf.random_normal([in_c,out_c]))
            self.act,self.d_act = act,d_act
            self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))
    def getw(self): return [self.w]
    def feedforward(self,input):
        self.input  = input 
        self.layer  = tf.matmul(input,self.w)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient):
        
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad = tf.matmul(tf.transpose(grad_part_3),tf.multiply(grad_part_1,grad_part_2))
        pass_on_grad = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return pass_on_grad,grad_update    

# Read the data traind and Test
train_data, train_label, test_data,test_label = get_data()

# hyper parameter
num_epoch = 101 
learning_rate = 0.00008
batch_size = 100
print_size = 10

proportion_rate = 1000
decay_rate = 0.08

beta1,beta2 = 0.9,0.999
adam_e = 0.00000001

# make layers
l1 = CNNLayer(7,3,50,tf_Relu,d_tf_Relu)
l2 = CNNLayer(5,50,50,tf_Relu,d_tf_Relu)

l3 = CNNLayer(5,50,150,tf_Relu,d_tf_Relu)
l4 = CNNLayer(5,150,150,tf_Relu,d_tf_Relu)

l5 = FNNLayer(8*8*150,512,tf_log,d_tf_log)
l6 = FNNLayer(512,512,tf_log,d_tf_log)
l7 = FNNLayer(512,10,tf_log,d_tf_log)

weight_list = l1.getw()+l2.getw()+l3.getw()+l4.getw()+l5.getw()+l6.getw()+l7.getw()

# make graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

layer1 = l1.feedforward(x,'SAME')
layer2 = l2.feedforward_mean(layer1,'SAME')

layer3 = l3.feedforward(layer2,'SAME')
layer4 = l4.feedforward_mean(layer3,'SAME')

layer5_Input = tf.reshape(layer4,[batch_size,-1])
layer5 = l5.feedforward(layer5_Input)
layer6 = l6.feedforward(layer5)
layer7 = l7.feedforward(layer6)

final_soft = tf_softmax(layer7)
cost = tf.reduce_sum(-1.0 * (y* tf.log(final_soft) + (1-y)*tf.log(1-final_soft)))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# auto train
# auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=weight_list)
auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost,var_list=weight_list)

# make session
# config = tf.ConfigProto(device_count = {'GPU': 0})
# with tf.Session(config=config) as sess: 
with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    train_total_cost,train_total_acc =0,0
    train_cost_overtime,train_acc_overtime = [],[]

    test_total_cost,test_total_acc = 0,0
    test_cost_overtime,test_acc_overtime = [],[]

    for iter in range(num_epoch):
        train_data,train_label = shuffle(train_data,train_label)

        # Train Set
        for current_batch_index in range(0,len(train_data),batch_size):
            
            current_batch = train_data[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_label[current_batch_index:current_batch_index+batch_size,:]

            sess_results = sess.run([cost,accuracy,correct_prediction,auto_train],feed_dict={x:current_batch,y:current_batch_label})
            # sess_results = sess.run([cost,accuracy,correct_prediction,grad_update],feed_dict={x:current_batch,y:current_batch_label,iter_variable_dil:iter})
            
            print("current iter:", iter,' Current batach : ',current_batch_index," current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_data),batch_size):

            current_batch = test_data[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = test_label[current_batch_index:current_batch_index+batch_size,:]

            sess_results = sess.run( [cost,accuracy,correct_prediction], feed_dict= {x:current_batch,y:current_batch_label,iter_variable_dil:iter})
            print("current iter:", iter,' Current batach : ',current_batch_index, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
            test_total_cost = test_total_cost + sess_results[0]
            test_total_acc = test_total_acc + sess_results[1]

        # store
        train_cost_overtime.append(train_total_cost/(len(train_data)/batch_size ) )
        train_acc_overtime.append(train_total_acc/(len(train_data)/batch_size ) )

        test_cost_overtime.append(test_total_cost/(len(test_data)/batch_size ) )
        test_acc_overtime.append(test_total_acc/(len(test_data)/batch_size ) )
        
        # print
        if iter%print_size == 0:
            print('\n=========')
            print("Avg Train Cost: ", train_cost_overtime[-1])
            print("Avg Train Acc: ", train_acc_overtime[-1])
            print("Avg Test Cost: ", test_cost_overtime[-1])
            print("Avg Test Acc: ", test_acc_overtime[-1])
            print('-----------')      

        train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0

    # plot and save
    plt.figure()
    plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r')
    plt.title('Train Cost over time')
    plt.show()

    plt.figure()
    plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='b')
    plt.title('Train Acc over time')
    plt.show()

    plt.figure()
    plt.plot(range(len(test_cost_overtime)),test_cost_overtime,color='y')
    plt.title('Test Cost over time')
    plt.show()

    plt.figure()
    plt.plot(range(len(test_acc_overtime)),test_acc_overtime,color='g')
    plt.title('Test Acc over time')
    plt.show()

# -- end code --