import os,sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from read_10_data import get_data
from sklearn.utils import shuffle
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
plt.style.use('ggplot')
ia.seed(1)
np.random.seed(789)
tf.set_random_seed(789)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.CropAndPad(percent=(-0.25, 0.25))
], random_order=True) # apply augmenters in random order

# activation functions here and there
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf_elu(x) + 1.0 

def tf_log(x): return tf.sigmoid(x) # dont use this 
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_tanh(x): return tf.nn.tanh(x) # dont use this 
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

def tf_atan(x): return tf.atan(x) # dont use this 
def d_tf_atan(x): return 1.0 / (1 + tf.square(x))

def tf_softmax(x): return tf.nn.softmax(x)


# === Get Data ===
train_images, train_labels, test_images,test_labels = get_data()

# === Augment Data ===
# train_images_augmented = seq.augment_images(train_images)
# train_images = np.concatenate((train_images,train_images_augmented),axis=0)
# train_labels = np.concatenate((train_labels,train_labels),axis=0)
# train_images,train_labels = shuffle(train_images,train_labels)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# === Hyper Parameter ===
num_epoch =  200
batch_size = 100
print_size = 1
shuffle_size = 1
divide_size = 5

proportion_rate = 1000
decay_rate = 0.08

learning_rate_dynamic = 0.01
momentum_rate = 0.9

drop_out_rate = 0.6
dynamic_drop_rate_change = 10000

dynamic_noise_rate = 0.9
one_channel = 128

# === Make Class ===
class CNNLayer():
    
    def __init__(self,kernel,in_c,out_c,act,d_act):
        # self.w = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c] ,stddev=5e-2)  )
        self.w = tf.Variable(tf.random_normal([kernel,kernel,in_c,out_c] ,stddev=5e-2)  )
        self.act = act
        self.d_act = d_act
    def getw(self): return self.w
    def getl2(self): return tf.nn.l2_loss(self.w)

    def feedforward(self,input,dropout=None,stride=1):
        self.input = input

        if not dropout==None:
            self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,stride,stride,1],padding='SAME')
            self.layerA = tf.nn.dropout(self.act(self.layer),dropout)
        else:
            self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,stride,stride,1],padding='SAME')
            self.layerA = self.act(self.layer)

        return self.layerA

# ---- Starting -----
l1_0 = CNNLayer(3,3,one_channel,tf_elu,d_tf_elu)

# ---- wide block 1 -----
block1_in,block1_out = one_channel,one_channel
l2_1 = CNNLayer(3,block1_in,block1_out,tf_elu,d_tf_elu)
l2_2 = CNNLayer(3,block1_out,block1_out,tf_elu,d_tf_elu)
l2_short = CNNLayer(3,block1_in,block1_out,tf_elu,d_tf_elu)

l3_1 = CNNLayer(1,block1_out,block1_out,tf_elu,d_tf_elu)
l3_2 = CNNLayer(1,block1_out,block1_out,tf_elu,d_tf_elu)

# ---- wide block 2 -----
block2_in,block2_out = one_channel,one_channel
l4_1 = CNNLayer(3,block2_in,block2_out,tf_elu,d_tf_elu)
l4_2 = CNNLayer(3,block2_out,block2_out,tf_elu,d_tf_elu)
l4_short = CNNLayer(3,block2_in,block2_out,tf_elu,d_tf_elu)

l5_1 = CNNLayer(1,block2_out,block2_out,tf_elu,d_tf_elu)
l5_2 = CNNLayer(1,block2_out,block2_out,tf_elu,d_tf_elu)

# ---- wide block 3 -----
block3_in,block3_out = one_channel,one_channel
l6_1 = CNNLayer(3,block3_in,block3_out,tf_elu,d_tf_elu)
l6_2 = CNNLayer(3,block3_out,block3_out,tf_elu,d_tf_elu)
l6_short = CNNLayer(3,block3_in,block3_out,tf_elu,d_tf_elu)

l7_1 = CNNLayer(1,block3_out,block3_out,tf_elu,d_tf_elu)
l7_2 = CNNLayer(1,block3_out,block3_out,tf_elu,d_tf_elu)

# ---- wide block 4 -----
block4_in,block4_out = one_channel,one_channel
l8_1 = CNNLayer(3,block4_in,block4_out,tf_elu,d_tf_elu)
l8_2 = CNNLayer(3,block4_out,block4_out,tf_elu,d_tf_elu)
l8_short = CNNLayer(3,block4_in,block4_out,tf_elu,d_tf_elu)

l9_1 = CNNLayer(1,block4_out,block4_out,tf_elu,d_tf_elu)
l9_2 = CNNLayer(1,block4_out,block4_out,tf_elu,d_tf_elu)

# ---- wide block 5 -----
block5_in,block5_out = one_channel,10
l10_1 = CNNLayer(3,block5_in,block5_out,tf_elu,d_tf_elu)
l10_2 = CNNLayer(3,block5_out,block5_out,tf_elu,d_tf_elu)
l10_short = CNNLayer(3,block5_in,block5_out,tf_elu,d_tf_elu)

l11_1 = CNNLayer(1,block5_out,block5_out,tf_elu,d_tf_elu)
l11_2 = CNNLayer(1,block5_out,block5_out,tf_elu,d_tf_elu)

# === Make graph ===
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32,[]) 
learning_rate = tf.placeholder(tf.float32,[]) 
is_train = tf.placeholder(tf.bool)
noise_rate = tf.placeholder(tf.float32)

# ---- start layer ----
layer1_0 = l1_0.feedforward(x)

# ---- wide block 1 -----
layer2_1 = l2_1.feedforward(layer1_0,keep_prob,2)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_short = l2_short.feedforward(layer1_0,stride=2)

# np.random.weibull(a=1,size=(batch_size,16,16,one_channel)) * noise_rate
if is_train is not None:
    layer2_add = tf.add(layer2_2,layer2_short) +layer2_short
else:
    layer2_add = tf.add(layer2_2,layer2_short) +layer2_short

layer3_1 = l3_1.feedforward(layer2_add,keep_prob)
layer3_2 = l3_2.feedforward(layer3_1)
layer3_add = tf.add(layer3_2,layer2_add) 

# ---- wide block 2 -----
layer4_1 = l4_1.feedforward(layer3_add,keep_prob,2)
layer4_2 = l4_2.feedforward(layer4_1)
layer4_short = l4_short.feedforward(layer3_add,stride=2)

if is_train is not None:
    layer4_add = tf.add(layer4_2,layer4_short)+layer4_short*np.random.randn(batch_size,8,8,one_channel) * noise_rate
else:
    layer4_add = tf.add(layer4_2,layer4_short)+layer4_short

layer5_1 = l5_1.feedforward(layer4_add,keep_prob)
layer5_2 = l5_2.feedforward(layer5_1)
layer5_add = tf.add(layer5_2,layer4_add)

# ---- wide block 3 -----
layer6_1 = l6_1.feedforward(layer5_add,keep_prob,2)
layer6_2 = l6_2.feedforward(layer6_1)
layer6_short = l6_short.feedforward(layer5_add,stride=2)

# *np.random.poisson(size=(batch_size,4,4,one_channel)) * noise_rate
if is_train is not None:
    layer6_add = tf.add(layer6_2,layer6_short)+layer6_short
else:
    layer6_add = tf.add(layer6_2,layer6_short)+layer6_short

layer7_1 = l7_1.feedforward(layer6_add,keep_prob)
layer7_2 = l7_2.feedforward(layer7_1)
layer7_add = tf.add(layer7_2,layer6_add)

# ---- wide block 4 -----
layer8_1 = l8_1.feedforward(layer7_add,keep_prob,2)
layer8_2 = l8_2.feedforward(layer8_1)
layer8_short = l8_short.feedforward(layer7_add,stride=2)

if is_train is not None:
    layer8_add = tf.add(layer8_2,layer8_short) + layer8_short*np.random.randn(batch_size,2,2,one_channel) * noise_rate
else:
    layer8_add = tf.add(layer8_2,layer8_short) + layer8_short

layer9_1 = l9_1.feedforward(layer8_add,keep_prob)
layer9_2 = l9_2.feedforward(layer9_1)
layer9_add = tf.add(layer9_2,layer8_add)

# ---- wide block 5 -----
layer10_1 = l10_1.feedforward(layer9_add,keep_prob,2)
layer10_2 = l10_2.feedforward(layer10_1)
layer10_short = l10_short.feedforward(layer9_add,stride=2)

# *np.random.randn(batch_size,1,1,10) * noise_rate
if is_train is not None:
    layer10_add = tf.add(layer10_2,layer10_short) + layer10_short
else:
    layer10_add = tf.add(layer10_2,layer10_short) + layer10_short

layer11_1 = l11_1.feedforward(layer10_add,keep_prob)
layer11_2 = l11_2.feedforward(layer11_1)
layer11_add = tf.add(layer11_2,layer10_add)

# --- final layer ----
final_soft = tf.reshape(layer11_add,[batch_size,-1])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= final_soft,labels=y))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- auto train ---
global_step = tf.Variable(0)
# auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost,global_step=global_step)
auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost)

# auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost + 0.05*regularizer) # This does not seem to work
# auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # This does not seem to work

# === Start the Session ===
plt.axis([0, 300, 0, 5])
plt.ion()
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

        # dynamic changes
        if iter == 250: 
            dynamic_noise_rate = dynamic_noise_rate*0.95
            drop_out_rate = drop_out_rate * 0.85
        elif iter == 175:
            dynamic_noise_rate = dynamic_noise_rate*0.95
            drop_out_rate = drop_out_rate * 0.85        
        elif iter == 75:
            dynamic_noise_rate = dynamic_noise_rate*1.1
            drop_out_rate = drop_out_rate * 1.1

        # Train Set
        for current_batch_index in range(0,int(len(train_images)/divide_size),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy,auto_train],feed_dict={x: current_batch, y: current_batch_label, keep_prob: drop_out_rate,learning_rate:learning_rate_dynamic,is_train:True,noise_rate:dynamic_noise_rate})
            sess_results[0] = sess_results[0] * 0.5
            print("current iter:", iter,' Drop Out Rate: %.3f'%drop_out_rate,' learning rate: %.3f'%learning_rate_dynamic ,' Current batach : ',current_batch_index," current cost: %.5f" % sess_results[0],' current acc: %.5f '%sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]
        # print('\n')

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
            current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy],feed_dict={x: current_batch, y: current_batch_label, keep_prob: 1.0,noise_rate:0.0})
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
            # test_images,test_labels = shuffle(test_images,test_labels)

        # dynamic learning rate
        # if iter == 200: 
        #     learning_rate_dynamic = learning_rate_dynamic * 0.1

        # increase the drop count
        # if test_acc_overtime[-1]<train_acc_overtime[-1]:
        #     count_drop = count_drop + 1

        # dynamic drop out decrease
        # if dynamic_drop_rate_change == count_drop :
        #     # drop_out_rate = drop_out_rate - 0.01
        #     drop_out_rate = drop_out_rate * 0.95
        #     count_drop = 0
            
        # redeclare
        train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0

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
    plt.figure()
    plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r',label="Train COT")
    plt.plot(range(len(train_cost_overtime)),test_cost_overtime,color='b',label='Test COT')
    plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='g',label="Train AOT")
    plt.plot(range(len(train_acc_overtime)),test_acc_overtime,color='y',label='Test AOT')
    plt.legend()
    plt.title('Results')
    plt.show()


# -- end code --