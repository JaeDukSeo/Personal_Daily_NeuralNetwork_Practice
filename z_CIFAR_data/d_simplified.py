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
def d_tf_elu(x): return tf_elu(x) + 1.0 

def tf_softmax(x): return tf.nn.softmax(x)

# === Make Class ===
class Convolution_Layer():
    
    def __init__(self,kernel,in_c,out_c,act,d_act):
        
        self.w = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c],stddev=5e-2))
        self.act,self.d_act = act,d_act

        self.m = tf.Variable(tf.zeros_like(self.w))
        self.v = tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,stride=1,padding='SAME',dropout=1.0):
        self.input = input
        self.layer  = tf.nn.dropout(tf.nn.conv2d(input,self.w,strides=[1,stride,stride,1],padding=padding),dropout)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient,stride=1):
        
        grad_part1 = gradient
        grad_part2 = self.d_act(self.layer)
        grad_part3 = self.input

        grad_middle = tf.multiply(grad_part1,grad_part2)

        grad_w = tf.nn.conv2d_backprop_filter(
            input = grad_part3,
            filter_sizes = self.w.shape,
            out_backprop = grad_middle,
            strides = [1,stride,stride,1],
            padding='SAME'
        )

        grad_input_shape = list(self.input.shape[1:])
        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes = [batch_size] + grad_input_shape,
            filter = self.w,
            out_backprop = grad_middle,
            strides = [1,stride,stride,1],
            padding='SAME'
        )

        updatew = []
        updatew.append(tf.assign(self.m,init_momentum_rate*self.m + init_learning_rate*grad_w))
        updatew.append(tf.assign(self.w,self.w - self.m))

        return grad_pass,updatew


# === Get Data ===
train_images, train_labels, test_images,test_labels = get_data()

# === Hyper Parameter ===
num_epoch =  200
batch_size = 100
print_size = 1
shuffle_size = 1
divide_size = 4

proportion_rate = 1000
decay_rate = 0.08

init_learning_rate = 0.00001
init_momentum_rate = 0.4
init_dropout_rate  = np.random.uniform(0.8,0.9)
init_noise_rate    = np.random.uniform(0.1,0.5)

one_channel = 128

# === Make Class ===
l1_1 = Convolution_Layer(3,3,one_channel,tf_elu,d_tf_elu)
l1_2 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l1_s = Convolution_Layer(1,3,one_channel,tf_elu,d_tf_elu)

l2_1 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l2_2 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l2_s = Convolution_Layer(1,one_channel,one_channel,tf_elu,d_tf_elu)

l3_1 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l3_2 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l3_s = Convolution_Layer(1,one_channel,one_channel,tf_elu,d_tf_elu)

l4_1 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l4_2 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l4_s = Convolution_Layer(1,one_channel,one_channel,tf_elu,d_tf_elu)

l5_1 = Convolution_Layer(3,one_channel,one_channel,tf_elu,d_tf_elu)
l5_2 = Convolution_Layer(3,one_channel,10,tf_elu,d_tf_elu)
l5_s = Convolution_Layer(1,one_channel,10,tf_elu,d_tf_elu)










# === Make graph ===
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

learning_rate = tf.placeholder(tf.float32,[]) 
momentum_rate = tf.placeholder(tf.float32,[]) 
dropout_rate  = tf.placeholder(tf.float32,[]) 

layer1_1 = l1_1.feedforward(x,2)
layer1_2 = l1_2.feedforward(layer1_1,dropout=dropout_rate)
layer1_s = l1_s.feedforward(x,2)
layer1_add = layer1_s + layer1_2

layer2_1 = l2_1.feedforward(layer1_add,2)
layer2_2 = l2_2.feedforward(layer2_1,dropout=dropout_rate)
layer2_s = l2_s.feedforward(layer1_add,2)
layer2_add = layer2_s + layer2_2

layer3_1 = l3_1.feedforward(layer2_add,2)
layer3_2 = l3_2.feedforward(layer3_1,dropout=dropout_rate)
layer3_s = l3_s.feedforward(layer2_add,2)
layer3_add = layer3_s + layer3_2

layer4_1 = l4_1.feedforward(layer3_add,2)
layer4_2 = l4_2.feedforward(layer4_1,dropout=dropout_rate)
layer4_s = l4_s.feedforward(layer3_add,2)
layer4_add = layer4_s + layer4_2

layer5_1 = l5_1.feedforward(layer4_add,2)
layer5_2 = l5_2.feedforward(layer5_1,dropout=dropout_rate)
layer5_s = l5_s.feedforward(layer4_add,2)
layer5_add = layer5_s + layer5_2

# --- final layer ---- adding clip by value
final_soft = tf.clip_by_value(tf_softmax(tf.reshape(layer5_add,[batch_size,-1])) ,1e-10,1.0)
cost = tf.reduce_mean( -1.0 * ( y * tf.log(final_soft)) + (1.0-y) * tf.log( 1.0-final_soft)   )
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- auto train ---
# auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost)

# --- manual back prop --- adding 1e-50 for numerical stability
error_soft = tf.reshape( final_soft -y,[batch_size,1,1,10])
grad5_s,grad5_sw = l5_s.backprop(error_soft,stride=2)
grad5_2,grad5_2w = l5_2.backprop(error_soft)
grad5_1,grad5_1w = l5_1.backprop(grad5_2,stride=2)

grad4_s,grad4_sw = l4_s.backprop(grad5_1+grad5_s,stride=2)
grad4_2,grad4_2w = l4_2.backprop(grad5_1+grad5_s)
grad4_1,grad4_1w = l4_1.backprop(grad4_2,stride=2)

grad3_s,grad3_sw = l3_s.backprop(grad4_1+grad4_s,stride=2)
grad3_2,grad3_2w = l3_2.backprop(grad4_1+grad4_s)
grad3_1,grad3_1w = l3_1.backprop(grad3_2,stride=2)

grad2_s,grad2_sw = l2_s.backprop(grad3_1+grad3_s,stride=2)
grad2_2,grad2_2w = l2_2.backprop(grad3_1+grad3_s)
grad2_1,grad2_1w = l2_1.backprop(grad2_2,stride=2)

grad1_s,grad1_sw = l1_s.backprop(grad2_1+grad2_s,stride=2)
grad1_2,grad1_2w = l1_2.backprop(grad2_1+grad2_s)
grad1_1,grad1_1w = l1_1.backprop(grad1_2,stride=2)

grad_update = grad5_sw + grad5_2w + grad5_1w + \
              grad4_sw + grad4_2w + grad4_1w + \
              grad3_sw + grad3_2w + grad3_1w + \
              grad2_sw + grad2_2w + grad2_1w + \
              grad1_sw + grad1_2w + grad1_1w 



# === Start the Session ===
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.Session() as sess: 

    # start the session 
    sess.run(tf.global_variables_initializer())
    train_total_cost,train_total_acc, test_total_cost,test_total_acc =0,0,0,0
    train_cost_overtime,train_acc_overtime,test_cost_overtime,test_acc_overtime = [],[],[],[]

    # Start the Epoch
    for iter in range(num_epoch):
        
        # dynamically change
        init_dropout_rate  = np.random.uniform(0.8,0.9)

        # Train Set
        for current_batch_index in range(0,int(len(train_images)/divide_size),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy,grad_update,final_soft,error_soft,correct_prediction],
                                feed_dict={x: current_batch, y: current_batch_label, learning_rate:init_learning_rate,momentum_rate:init_momentum_rate,dropout_rate:init_dropout_rate})

            print("current iter:", iter,' Drop Out Rate: %.3f'%init_dropout_rate,' learning rate: %.3f'%init_learning_rate ,
                ' Current batach : ',current_batch_index," current cost: %.5f" % sess_results[0],' current acc: %.5f '%sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
            current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy],feed_dict={x: current_batch, y: current_batch_label,dropout_rate:1.0})

            print("Test Image Current iter:", iter,' Drop Out Rate: %.3f'%init_dropout_rate,' learning rate: %.3f'%init_learning_rate,
                 ' Current batach : ',current_batch_index, " current cost: %.5f" % sess_results[0],' current acc: %.5f '%sess_results[1], end='\r')
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