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

def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_acrtan(x): return tf.atan(x)
def d_tf_arctan(x): return 1/(1+tf.square(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_softmax(x): return tf.nn.softmax(x)

# make class
class CNNLayer():
    
    def __init__(self,kernel,in_c,out_c,act,d_act):
        with tf.device('/cpu:0'):
            self.w = tf.Variable(tf.random_normal([kernel,kernel,in_c,out_c]))
            self.act,self.d_act = act,d_act
            self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))
    def getw(self): return self.w
    def feedforward(self,input,padding,resinput=None):
        
        self.input  = input 
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding=padding)
        self.layerA = self.act(self.layer)

        if resinput==None:
            self.layerA = self.act(self.layer)
        else: 
            self.layerA = self.act(self.layer)+resinput
            
        return self.layerA

    def feedforward2(self,input,padding,resinput=None):
        self.input  = input 
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding=padding)
        if resinput==None:
            self.layerA = self.act(self.layer)
        else: 
            self.layerA = self.act(self.layer)+resinput
            
        return self.layerA

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
    def getw(self): return self.w
    def feedforward(self,input,resinput=None):
        self.input  = input 
        self.layer  = tf.matmul(input,self.w)
        self.layerA = self.act(self.layer)
        if resinput==None:
            self.layerA = self.act(self.layer)
        else: 
            self.layerA = self.act(self.layer)+resinput
            
        return self.layerA
    
    def feedforward2(self,input):
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
learning_rate = 0.001
batch_size = 100
print_size = 3

proportion_rate = 1000
decay_rate = 0.08

beta1,beta2 = 0.9,0.999
adam_e = 0.00000001
change_channel = 10
change_size = 32
fully_connected_neuron = 512

# make class 
Cl1 = CNNLayer(11,3,change_channel,tf_Relu,d_tf_Relu)
Cl2 = CNNLayer(9, change_channel,change_channel,tf_Relu,d_tf_Relu)
Cl3 = CNNLayer(7, change_channel,change_channel,tf_Relu,d_tf_Relu)
Cl4 = CNNLayer(5, change_channel,change_channel,tf_Relu,d_tf_Relu)
Cl5 = CNNLayer(3, change_channel,change_channel,tf_Relu,d_tf_Relu)
Cl6 = CNNLayer(1, change_channel,change_channel,tf_Relu,d_tf_Relu)

Fl1 = FNNLayer(change_size*change_size*
               change_channel,fully_connected_neuron,tf_log,d_tf_log)
Fl2 = FNNLayer(fully_connected_neuron,fully_connected_neuron,tf_log,d_tf_log)
Fl3 = FNNLayer(fully_connected_neuron,fully_connected_neuron,tf_log,d_tf_log)
Fl4 = FNNLayer(fully_connected_neuron,10,tf_log,d_tf_log)

weights = [
        Cl1.getw(),Cl2.getw(),
        Cl3.getw(),Cl4.getw(),
        Cl5.getw(),Cl6.getw(),
        Fl1.getw(),Fl2.getw(),
        Fl3.getw(),Fl4.getw()]

# make graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

layer1 = Cl1.feedforward(x,'SAME')
layer2 = Cl2.feedforward(layer1,'SAME',layer1)
layer3 = Cl3.feedforward(layer2,'SAME',layer2 )
layer4 = Cl4.feedforward(layer3,'SAME',layer3 )

layer5 = Cl5.feedforward(layer4,'SAME',layer4 )
layer6 = Cl6.feedforward(layer5,'SAME' )

Flayer5_Input = tf.reshape(layer6,[batch_size,-1])
Flayer5 = Fl1.feedforward(Flayer5_Input)
Flayer6 = Fl2.feedforward(Flayer5,Flayer5)
Flayer7 = Fl3.feedforward(Flayer6,Flayer6)
Flayer8 = Fl4.feedforward(Flayer7)

final_soft = tf_softmax(Flayer8)
cost = tf.reduce_sum(-1.0 * (y* tf.log(final_soft) + (1-y)*tf.log(1-final_soft)))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# auto train
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=weights)

# back propagation
grad_8,grad_8w = Fl4.backprop(final_soft-y)
grad_7,grad_7w = Fl3.backprop(grad_8+decay_propotoin_rate*(grad_8))
grad_6,grad_6w = Fl2.backprop(grad_7+decay_propotoin_rate*(grad_8+grad_7))
grad_5,grad_5w = Fl1.backprop(grad_6+decay_propotoin_rate*(grad_8+grad_7+grad_6))

grad_4_Input = tf.reshape(grad_5,[batch_size,change_size,change_size,change_channel])
grad_4,grad_4w = Cl4.backprop(grad_4_Input+decay_propotoin_rate*(grad_4_Input),'SAME')
grad_3,grad_3w = Cl3.backprop(grad_4+decay_propotoin_rate*(grad_4_Input+grad_4),'SAME')
grad_2,grad_2w = Cl2.backprop(grad_3+decay_propotoin_rate*(grad_4_Input+grad_4+grad_3),'SAME')
grad_1,grad_1w = Cl1.backprop(grad_2+decay_propotoin_rate*(grad_4_Input+grad_4+grad_3+grad_2),'SAME')
grad_update = grad_8w+grad_7w+grad_6w+grad_5w+grad_4w+grad_3w+grad_2w+grad_1w

# make session
with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    train_total_cost,train_total_acc =0,0
    train_cost_overtime,train_acc_overtime = [],[]

    test_total_cost,test_total_acc = 0,0
    test_cost_overtime,test_acc_overtime = [],[]

    for iter in range(num_epoch):
        
        # train_data,train_label = shuffle(train_data,train_label)

        # train
        for current_batch_index in range(0,len(train_data),batch_size):
            
            current_batch = train_data[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_label[current_batch_index:current_batch_index+batch_size,:]

            # sess_results = sess.run([cost,accuracy,correct_prediction,auto_train],feed_dict={x:current_batch,y:current_batch_label})
            sess_results = sess.run([cost,accuracy,correct_prediction,grad_update],feed_dict={x:current_batch,y:current_batch_label,iter_variable_dil:iter})
            
            print("current iter:", iter,' Current batach : ',current_batch_index," current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test batch
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
    # plt.savefig('Train Cost over time')
    plt.show()

    plt.figure()
    plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='b')
    # plt.savefig('Train Acc over time')
    plt.show()

    plt.figure()
    plt.plot(range(len(test_cost_overtime)),test_cost_overtime,color='y')
    # plt.savefig('Test Cost over time')
    plt.show()

    plt.figure()
    plt.plot(range(len(test_acc_overtime)),test_acc_overtime,color='g')
    # plt.savefig('Test Acc over time')
    plt.show()

# -- end code --