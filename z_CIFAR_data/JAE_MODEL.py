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
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
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
train_images_augmented = seq.augment_images(train_images)
train_images = np.concatenate((train_images,train_images_augmented),axis=0)
train_labels = np.concatenate((train_labels,train_labels),axis=0)
train_images,train_labels = shuffle(train_images,train_labels)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# === Hyper ===
num_epoch =  300
batch_size = 100
print_size = 1
shuffle_size = 10
divide_size = 4

proportion_rate = 1000
decay_rate = 0.08

learning_rate = 0.0006
momentum_rate = 0.7















# =========== Layer Class ===========
# === Convolutional Layer ===
class CNNLayer():
      
  def __init__(self,kernel,in_c,out_c,act,d_act):
    with tf.device('/cpu:0'):
      self.w = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c],stddev=0.05,mean=0.0))
      self.act,self.d_act = act,d_act
      self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))
  def getw(self): return [self.w]
  def reg(self): return tf.nn.l2_loss(self.w)

  def feedforward(self,input):
    self.input = input
    self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME')
    self.layerA = self.act(self.layer)
    return self.layerA

  def feedforward_res(self,input,resinput):
    self.input = input
    self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME')
    self.layerA = self.act(self.layer) + resinput
    return self.layerA

  def feedforward_dropout(self,input,droprate):
    self.input = input
    self.layer = tf.nn.dropout(tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME'),droprate)
    self.layerA = self.act(self.layer)
    return self.layerA

  def feedforward_avg(self,input,droprate):
    self.input = input
    self.layer =  tf.nn.dropout(tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME'),droprate)
    self.layerA = self.act(self.layer)
    self.layerMean = tf.nn.avg_pool(self.layerA, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return self.layerMean 












# === Make Layers ===
l0_1 = CNNLayer(7,3,48,  tf_elu,d_tf_elu)
l0_2 = CNNLayer(1,48,48,tf_elu,d_tf_elu)
l0_3 = CNNLayer(5,48,48,tf_elu,d_tf_elu)
l0_4 = CNNLayer(1,48,96,tf_elu,d_tf_elu)

l1_1 = CNNLayer(3,96,96,tf_elu,d_tf_elu)
l1_2 = CNNLayer(1,96,96,tf_elu,d_tf_elu)
l1_3 = CNNLayer(2,96,96,tf_elu,d_tf_elu)
l1_4 = CNNLayer(1,96,192,tf_elu,d_tf_elu)

l2_1 = CNNLayer(2,192,192,tf_elu,d_tf_elu)
l2_2 = CNNLayer(1,192,192,tf_elu,d_tf_elu)
l2_3 = CNNLayer(1,192,192,tf_elu,d_tf_elu)
l2_4 = CNNLayer(1,192,10,tf_elu,d_tf_elu)


















# === Make Graph ===
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

droprate1 = tf.placeholder(shape=[],dtype=tf.float32)
droprate2 = tf.placeholder(shape=[],dtype=tf.float32)
droprate3 = tf.placeholder(shape=[],dtype=tf.float32)
droprate4 = tf.placeholder(shape=[],dtype=tf.float32)

iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

layer0_1 = l0_1.feedforward(x)
layer0_2 = l0_2.feedforward(layer0_1)
layer0_3 = l0_3.feedforward(layer0_2)
layer0_4 = l0_4.feedforward(layer0_3)

layer1_1 = l1_1.feedforward(layer0_4)
layer1_2 = l1_2.feedforward(layer1_1)
layer1_3 = l1_3.feedforward(layer1_2)
layer1_4 = l1_4.feedforward(layer1_3)

layer2_1 = l2_1.feedforward(layer1_4)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_3 = l2_3.feedforward(layer2_2)
layer2_4 = l2_4.feedforward(layer2_3)

global_norm = tf.reduce_mean(layer2_4,[1,2])
final_soft = tf_softmax(global_norm)
cost = tf.reduce_sum(-1.0 * (y*tf.log(final_soft) + (1.0-y)*tf.log(1.0-final_soft)))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- auto train ---
auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost)






























# === Start the Session ===
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
            sess_results = sess.run([cost,accuracy,correct_prediction,auto_train],feed_dict={x:current_batch,
            y:current_batch_label,iter_variable_dil:iter,droprate1:1.0,droprate2:1.0,droprate3:1.0,droprate4:1.0})
            print("current iter:", iter,' Current batach : ',current_batch_index," current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
          current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
          current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
          sess_results = sess.run( [cost,accuracy,correct_prediction], feed_dict= {x:current_batch,y:current_batch_label
          ,droprate1:1.0,droprate2:1.0,droprate3:1.0,droprate4:1.0})
          print("\t\t\tTest Image Current iter:", iter,' Current batach : ',current_batch_index, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
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

        if iter%shuffle_size ==  0: 
          print("\n==== shuffling iter: =====",iter," \n")
          train_images,train_labels = shuffle(train_images,train_labels)
          test_images,test_labels = shuffle(test_images,test_labels)
          
            
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