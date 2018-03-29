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

        updatew = []

        return 3

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

learning_rate_dynamic = 0.01
momentum_rate = 0.9

drop_out_rate = np.random.uniform(0,1)
dynamic_noise_rate = 0.9

one_channel = 128

# === Make Class ===
layer1_1 = 
layer1_2 = 
layer1_s = 






# === Make graph ===
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

learning_rate = tf.placeholder(tf.float32,[]) 
noise_rate = tf.placeholder(tf.float32)









# --- final layer ----
final_soft = tf.reshape(layer5_2,[batch_size,-1])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= final_soft,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- auto train ---
global_step = tf.Variable(0)
auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost,global_step=global_step)


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
        
        if iter < 100:
            drop_out_rate1 = np.random.uniform(0.8,0.9)
            drop_out_rate2 = np.random.uniform(0.7,0.8)
            drop_out_rate3 = np.random.uniform(0.7,0.9)
            drop_out_rate4 = np.random.uniform(0.9,0.98)
            drop_out_rate5 = np.random.uniform(0.8,0.93)
        elif iter < 200 and iter > 100:
            drop_out_rate1 = np.random.uniform(0.7,0.8)
            drop_out_rate2 = np.random.uniform(0.8,0.84)
            drop_out_rate3 = np.random.uniform(0.8,0.9)
            drop_out_rate4 = np.random.uniform(0.8,0.88)
            drop_out_rate5 = np.random.uniform(0.9,0.93)
            

        # Train Set
        for current_batch_index in range(0,int(len(train_images)/divide_size),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy,auto_train],feed_dict={x: current_batch, y: current_batch_label, 
                                                                keep_prob1: drop_out_rate1,
                                                                keep_prob2: drop_out_rate2,
                                                                keep_prob3: drop_out_rate3,
                                                                keep_prob4: drop_out_rate4,
                                                                keep_prob5: drop_out_rate5,
                                                                learning_rate:learning_rate_dynamic,noise_rate:dynamic_noise_rate})
            sess_results[0] = sess_results[0] * 0.5
            print("current iter:", iter,' Drop Out Rate: %.3f'%drop_out_rate,' learning rate: %.3f'%learning_rate_dynamic ,' Current batach : ',current_batch_index," current cost: %.5f" % sess_results[0],' current acc: %.5f '%sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
            current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy],feed_dict={x: current_batch, y: current_batch_label,
                                                                            keep_prob1: 1.0,
                                                                keep_prob2: 1.0,
                                                                keep_prob3: 1.0,
                                                                keep_prob4: 1.0,
                                                                keep_prob5: 1.0, noise_rate:0.0})
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