import os,sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from read_10_data import get_data
from sklearn.utils import shuffle
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)
np.random.seed(789)
tf.set_random_seed(789)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.CropAndPad(percent=(-0.25, 0.25))
], random_order=True) # apply augmenters in random order

# activation functions here and there
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf_elu(x) + 1.0 

def tf_softmax(x): return tf.nn.softmax(x)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

def tf_atan(x): return tf.atan(x)
def d_tf_atan(x): return 1.0 / (1 + tf.square(x))

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
num_epoch =  100
batch_size = 100
print_size = 1
shuffle_size = 2
divide_size = 5

proportion_rate = 1000
decay_rate = 0.08

learning_rate_dynamic = 0.001
momentum_rate = 0.9


# === Make Class ===
class CNNLayer():
    def __init__(self,kernel,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c] ,stddev=1e-8)  )
        self.act = act
        self.d_act = d_act
    def getw(self): return self.w
    def getl2(self): return tf.nn.l2_loss(self.w)
    def feedforward(self,input,dropout=None,stride=1):
        self.input = input
        if not dropout==None:
            self.layer = tf.nn.dropout(tf.nn.conv2d(self.input,self.w,strides=[1,stride,stride,1],padding='SAME'),dropout)
            self.layerA = self.act(self.layer)
            self.layerA = self.layerA
        else:
            self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,stride,stride,1],padding='SAME')
            self.layerA = self.act(self.layer)
            self.layerA = self.layerA
        return self.layerA

# ---- Starting -----
l1_0 = CNNLayer(3,3,16,tf_elu,d_tf_elu)

# ---- wide block 1 -----
block1_in,block1_out = 16,64
l2_1 = CNNLayer(3,block1_in,block1_out,tf_elu,d_tf_elu)
l2_2 = CNNLayer(3,block1_out,block1_out,tf_elu,d_tf_elu)
l2_short = CNNLayer(1,block1_in,block1_out,tf_elu,d_tf_elu)

l3_1 = CNNLayer(3,block1_out,block1_out,tf_elu,d_tf_elu)
l3_2 = CNNLayer(1,block1_out,block1_out,tf_elu,d_tf_elu)

# ---- wide block 2 -----
block2_in,block2_out = 64,4
l4_1 = CNNLayer(3,block2_in,block2_out,tf_elu,d_tf_elu)
l4_2 = CNNLayer(3,block2_out,block2_out,tf_elu,d_tf_elu)
l4_short = CNNLayer(1,block2_in,block2_out,tf_elu,d_tf_elu)

l5_1 = CNNLayer(3,block2_out,block2_out,tf_elu,d_tf_elu)
l5_2 = CNNLayer(1,block2_out,block2_out,tf_elu,d_tf_elu)

# ---- wide block 3 -----
block3_in,block3_out = 4,512
l6_1 = CNNLayer(3,block3_in,block3_out,tf_elu,d_tf_elu)
l6_2 = CNNLayer(3,block3_out,block3_out,tf_elu,d_tf_elu)
l6_short = CNNLayer(1,block3_in,block3_out,tf_elu,d_tf_elu)

l7_1 = CNNLayer(3,block3_out,block3_out,tf_elu,d_tf_elu)
l7_2 = CNNLayer(1,block3_out,block3_out,tf_elu,d_tf_elu)

# ---- wide block 4 -----
block4_in,block4_out = 512,4
l8_1 = CNNLayer(3,block4_in,block4_out,tf_elu,d_tf_elu)
l8_2 = CNNLayer(3,block4_out,block4_out,tf_elu,d_tf_elu)
l8_short = CNNLayer(1,block4_in,block4_out,tf_elu,d_tf_elu)

l9_1 = CNNLayer(3,block4_out,block4_out,tf_elu,d_tf_elu)
l9_2 = CNNLayer(1,block4_out,block4_out,tf_elu,d_tf_elu)

# ---- wide block 5 -----
block5_in,block5_out = 4,10
l10_1 = CNNLayer(3,block5_in,block5_out,tf_elu,d_tf_elu)
l10_2 = CNNLayer(3,block5_out,block5_out,tf_elu,d_tf_elu)
l10_short = CNNLayer(1,block5_in,block5_out,tf_elu,d_tf_elu)

l11_1 = CNNLayer(3,block5_out,block5_out,tf_elu,d_tf_elu)
l11_2 = CNNLayer(1,block5_out,block5_out,tf_elu,d_tf_elu)



# === Make graph ===
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32,[]) 
learning_rate = tf.placeholder(tf.float32,[]) 

# --- starting layer ----
layer1_0 = l1_0.feedforward(x)

# ---- wide block 1 -----
layer2_1 = l2_1.feedforward(layer1_0,keep_prob,2)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_short = l2_short.feedforward(layer1_0,stride=2)
layer2_add = tf.add(layer2_2,layer2_short)

layer3_1 = l3_1.feedforward(layer2_add,keep_prob)
layer3_2 = l3_2.feedforward(layer3_1)
layer3_add = tf.add(layer3_2,layer2_add)

# ---- wide block 2 -----
layer4_1 = l4_1.feedforward(layer3_add,keep_prob,2)
layer4_2 = l4_2.feedforward(layer4_1)
layer4_short = l4_short.feedforward(layer3_add,stride=2)
layer4_add = tf.add(layer4_2,layer4_short)

layer5_1 = l5_1.feedforward(layer4_add,keep_prob)
layer5_2 = l5_2.feedforward(layer5_1)
layer5_add = tf.add(layer5_2,layer4_add)

# ---- wide block 3 -----
layer6_1 = l6_1.feedforward(layer5_add,keep_prob,2)
layer6_2 = l6_2.feedforward(layer6_1)
layer6_short = l6_short.feedforward(layer5_add,stride=2)
layer6_add = tf.add(layer6_2,layer6_short)

layer7_1 = l7_1.feedforward(layer6_add,keep_prob)
layer7_2 = l7_2.feedforward(layer7_1)
layer7_add = tf.add(layer7_2,layer6_add)

# ---- wide block 4 -----
layer8_1 = l8_1.feedforward(layer7_add,keep_prob,2)
layer8_2 = l8_2.feedforward(layer8_1)
layer8_short = l8_short.feedforward(layer7_add,stride=2)
layer8_add = tf.add(layer8_2,layer8_short)

layer9_1 = l9_1.feedforward(layer8_add,keep_prob)
layer9_2 = l9_2.feedforward(layer9_1)
layer9_add = tf.add(layer9_2,layer8_add)

# ---- wide block 5 -----
layer10_1 = l10_1.feedforward(layer9_add,keep_prob,2)
layer10_2 = l10_2.feedforward(layer10_1)
layer10_short = l10_short.feedforward(layer9_add,stride=2)
layer10_add = tf.add(layer10_2,layer10_short)

layer11_1 = l11_1.feedforward(layer10_add,keep_prob)
layer11_2 = l11_2.feedforward(layer11_1)
layer11_add = tf.add(layer11_2,layer10_add)

# --- final layer ----
final_soft = tf_softmax(tf.reshape(layer11_add,[batch_size,-1]))
cost = tf.reduce_mean(-1.0 * (y*tf.log(final_soft) + (1.0-y)*tf.log(1.0-final_soft)))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- auto train ---
auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost,global_step=tf.Variable(0))


# --- space for manual back prop ---


# --- space for manual back prop ---





# === Start the Session ===
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0,allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.Session() as sess: 

  sess.run(tf.global_variables_initializer())

  train_total_cost,train_total_acc =0,0
  train_cost_overtime,train_acc_overtime = [],[]

  test_total_cost,test_total_acc = 0,0
  test_cost_overtime,test_acc_overtime = [],[]

  for iter in range(num_epoch):

        # Train Set
        for current_batch_index in range(0,int(len(train_images)/divide_size),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_labels[current_batch_index:current_batch_index+batch_size,:]

            if iter == 70: 
                learning_rate_dynamic = learning_rate_dynamic * 0.1

            sess_results =  sess.run([cost,accuracy,auto_train],feed_dict={x: current_batch, y: current_batch_label, keep_prob: 0.9,learning_rate:learning_rate_dynamic})

            print("current iter:", iter,' Current batach : ',current_batch_index,' current acc: ',sess_results[1]," current cost: ", sess_results[0], end='\n')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
          current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
          current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
          sess_results =  sess.run([cost,accuracy],feed_dict={x: current_batch, y: current_batch_label, keep_prob: 1.0})
          print("\t\t\tTest Image Current iter:", iter,' Current batach : ',current_batch_index,' current acc: ',sess_results[1], " current cost: ", sess_results[0], end='\r')
          test_total_cost = test_total_cost + sess_results[0]
          test_total_acc = test_total_acc + sess_results[1]

        # store
        train_cost_overtime.append(train_total_cost/(len(train_images)/divide_size/batch_size ) ) 
        train_acc_overtime.append(train_total_acc /(len(train_images)/divide_size/batch_size )  )
        test_cost_overtime.append(test_total_cost/(len(test_images)/batch_size ) )
        test_acc_overtime.append(test_total_acc/(len(test_images)/batch_size ) )
        
        # print
        if iter%print_size == 0:
            print('\n=========')
            print("Avg Train Cost: ", train_cost_overtime[-1])
            print("Avg Train Acc: ", train_acc_overtime[-1])
            print("Avg Test Cost: ", test_cost_overtime[-1])
            print("Avg Test Acc: ", test_acc_overtime[-1])
            print('-----------')      

        # shuffle
        if iter%shuffle_size ==  0: 
          print("\n==== shuffling iter: =====",iter," \n")
          train_images,train_labels = shuffle(train_images,train_labels)
          test_images,test_labels = shuffle(test_images,test_labels)
          
        # redeclare
        train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0

  # plot and save
  plt.figure()
  plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r',label="Train COT")
  plt.plot(range(len(train_cost_overtime)),test_cost_overtime,color='b',label='Test COT')
  plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='y',label="Train Acc")
  plt.plot(range(len(train_acc_overtime)),test_acc_overtime,color='g',label='Test Acc')
  plt.legend()
  plt.title('Training Results')
  plt.show()



# -- end code --