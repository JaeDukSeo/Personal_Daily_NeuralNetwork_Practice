import os,sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from read_10_data import get_data
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(789)
tf.set_random_seed(789)

# activation functions here and there
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf_elu(x) + keep_prob 

def tf_log(x): return tf.sigmoid(x) # dont use this 
def d_tf_log(x): return tf_log(x) * (keep_prob - tf_log(x))

def tf_tanh(x): return tf.nn.tanh(x) # dont use this 
def d_tf_tanh(x): return keep_prob - tf.square(tf_tanh(x))

def tf_atan(x): return tf.atan(x) # dont use this 
def d_tf_atan(x): return keep_prob / (1 + tf.square(x))

def tf_softmax(x): return tf.nn.softmax(x)

# === Make Class ===
class Convolution_Layer():
    
    def __init__(self,kernel,in_c,out_c,act,d_act):
        
        self.w = tf.Variable(tf.truncated_normal([kernel,kernel,inc_c,out_c],atddev=5e-2))
        self.act,self.d_act = act,d_act

        self.m = tf.Variable(tf.zeros_like(self.w))
        self.v = tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,stride=1,padding='SAME',dropout_rate=1.0):
        self.input = input
        self.layer  = tf.nn.dropout(tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding),dropout_rate)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient,stride=1):
        
        grad_part1 = gradient
        grad_part2 = self.d_act(self.layer)
        grad_part3 = self.input

        grad_middle = tf.nn.multiply(grad_part1,grad_part2)

        grad_w = tf.nn.conv2d_backprop_filter(
            input = grad_part3,
            filter_sizes = self.w.shape,
            out_backprop = grad_middle,
            strides = [1,stride,stride,1],
            padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = self.input.shape,
            filter = self.w,
            out_backprop = grad_middle,
            strides = [1,stride,stride,1],
            padding='SAME'
        )

        updatew = []
        updatew.append(tf.assign(self.m,alpha*self.m + learning_rate*grad_w))

        return grad_pass,updatew

# === Get Data ===
train_images, train_labels, test_images,test_labels = get_data()

# === Hyper Parameter ===
num_epoch =  200
batch_size = 500
print_size = 1
shuffle_size = 1
divide_size = 4

proportion_rate = 1000
decay_rate = 0.08

init_learning_rate = 0.01
init_momentum_rate = 0.9

drop_out_rate = np.random.uniform(0.8,0.9)
dynamic_noise_rate = 0.9

one_channel = 56

# === Make Class ===
l1_1 = Convolution_Layer(3,3,one_channel,tf_elu,d_tf_elu)
l1_2 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l1_s = Convolution_Layer(1,3,one_channel,tf_elu,d_tf_elu)

l2_1 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l2_2 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l2_s = Convolution_Layer(1,3,one_channel,tf_elu,d_tf_elu)

l3_1 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l3_2 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l3_s = Convolution_Layer(1,3,one_channel,tf_elu,d_tf_elu)

l4_1 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l4_2 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l4_s = Convolution_Layer(1,3,one_channel,tf_elu,d_tf_elu)

l5_1 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l5_2 = Convolution_Layer(3,one_channel,10,tf_elu,d_tf_elu)
l5_s = Convolution_Layer(1,3,10,tf_elu,d_tf_elu)










# === Make graph ===
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

learning_rate = tf.placeholder(tf.float32,[]) 
momentum_rate = tf.placeholder(tf.float32,[]) 

layer1_1 = l1_1.feedforward(x,2)
layer1_2 = l1_2.feedforward(layer1_1)
layer1_s = l1_s.feedforward(x,2)
layer1_add = layer1_s + layer1_2

layer2_1 = l2_1.feedforward(layer1_add,2)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_s = l2_s.feedforward(layer1_add,2)
layer2_add = layer2_s + layer2_2

layer3_1 = l3_1.feedforward(layer2_add,2)
layer3_2 = l3_2.feedforward(layer3_1)
layer3_s = l3_s.feedforward(layer2_add,2)
layer3_add = layer3_s + layer3_2

layer4_1 = l4_1.feedforward(layer3_add,2)
layer4_2 = l4_2.feedforward(layer4_1)
layer4_s = l4_s.feedforward(layer3_add,2)
layer4_add = layer4_s + layer4_2

layer5_1 = l5_1.feedforward(layer4_add,2)
layer5_2 = l5_2.feedforward(layer5_1)
layer5_s = l5_s.feedforward(layer4_add,2)
layer5_add = layer5_s + layer5_2

# --- final layer ----
final_soft = tf.reshape(layer5_add,[batch_size,-1])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= final_soft,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- auto train ---
global_step = tf.Variable(0)
auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost,global_step=global_step)



sys.exit()















# === Start the Session ===
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.Session() as sess: 

    # start the session 
    sess.run(tf.global_variables_initializer())
    train_total_cost,train_total_acc, test_total_cost,test_total_acc =0,0,0,0
    train_cost_overtime,train_acc_overtime,test_cost_overtime,test_acc_overtime = [],[],[],[]
    count_drop = 0

    # Start the Epoch
    for iter in range(num_epoch):
        
        # Train Set
        for current_batch_index in range(0,int(len(train_images)/divide_size),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy,auto_train],feed_dict={x: current_batch, y: current_batch_label, learning_rate:learning_rate_dynamic})
            sess_results[0] = sess_results[0] * 0.5
            print("current iter:", iter,' Drop Out Rate: %.3f'%drop_out_rate,' learning rate: %.3f'%learning_rate_dynamic ,' Current batach : ',current_batch_index," current cost: %.5f" % sess_results[0],' current acc: %.5f '%sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
            current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy],feed_dict={x: current_batch, y: current_batch_label})
            sess_results[0] = sess_results[0] * 0.5
            print("Test Image Current iter:", iter,' Drop Out Rate: %.3f'%drop_out_rate,' learning rate: %.3f'%learning_rate_dynamic,' Current batach : ',current_batch_index, " current cost: %.5f" % sess_results[0],' current acc: %.5f '%sess_results[1], end='\r')
            test_total_cost = test_total_cost + sess_results[0]
            test_total_acc = test_total_acc + sess_results[1]

        # store
        train_cost_overtime.append(train_total_cost/(len(train_images)/divide_size/batch_size )  ) 
        train_acc_overtime.append(train_total_acc /(len(train_images)/divide_size/batch_size )  )
        test_cost_overtime.append(test_total_cost/(len(test_images)/batch_size ))
        test_acc_overtime.append(test_total_acc/(len(test_images)/batch_size ))
            
        # print
        if iter%print_size == 0:
            print('\n\n==== Current Iter :', iter,' Average Results =====')
            print("Avg Train Cost: %.5f"% train_cost_overtime[-1])
            print("Avg Train Acc:  %.5f"% train_acc_overtime[-1])
            print("Avg Test Cost:  %.5f"% test_cost_overtime[-1])
            print("Avg Test Acc:   %.5f"% test_acc_overtime[-1])
            print('=================================')      

        # shuffle 
        if iter%shuffle_size ==  0: 
            print("==== shuffling iter: ",iter," =======\n")
            train_images,train_labels = shuffle(train_images,train_labels)

        # redeclare
        train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0

        # real time ploting
        if iter > 0: plt.clf()
        plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r',label="Train COT")
        plt.plot(range(len(train_cost_overtime)),test_cost_overtime,color='b',label='Test COT')
        plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='g',label="Train AOT")
        plt.plot(range(len(train_acc_overtime)),test_acc_overtime,color='y',label='Test AOT')
        plt.legend()
        plt.axis('auto')
        plt.title('Results')
        plt.pause(0.1)

    # plot and save
    plt.clf()
    plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r',label="Train COT")
    plt.plot(range(len(train_cost_overtime)),test_cost_overtime,color='b',label='Test COT')
    plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='g',label="Train AOT")
    plt.plot(range(len(train_acc_overtime)),test_acc_overtime,color='y',label='Test AOT')
    plt.legend()
    plt.title('Results')
    plt.show()


# -- end code --