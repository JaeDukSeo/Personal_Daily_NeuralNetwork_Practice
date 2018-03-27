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
    iaa.Fliplr(0.7), # horizontal flips
    iaa.Flipud(0.7), # horizontal flips
    iaa.Crop(percent=(0, 0.3)), # random crops
    iaa.Sometimes(0.1,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # iaa.ContrastNormalization((0.75, 1.5)),
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # iaa.Affine(
    #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     rotate=(-25, 25),
    #     shear=(-8, 8)
    # )
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


# train_images[:,:,:,0] = (train_images[:,:,:,0]-train_images[:,:,:,0].min(axis=0))/(train_images[:,:,:,0].max(axis=0) -train_images[:,:,:,0].min(axis=0)  )
# train_images[:,:,:,1] = (train_images[:,:,:,1]-train_images[:,:,:,1].min(axis=0))/(train_images[:,:,:,1].max(axis=0) -train_images[:,:,:,1].min(axis=0)  )
# train_images[:,:,:,2] = (train_images[:,:,:,2]-train_images[:,:,:,2].min(axis=0))/(train_images[:,:,:,2].max(axis=0) -train_images[:,:,:,2].min(axis=0)  )

train_images_augmented[:,:,:,0] = (train_images_augmented[:,:,:,0]-train_images_augmented[:,:,:,0].min(axis=0))/(train_images_augmented[:,:,:,0].max(axis=0) -train_images_augmented[:,:,:,0].min(axis=0)  )
train_images_augmented[:,:,:,1] = (train_images_augmented[:,:,:,1]-train_images_augmented[:,:,:,1].min(axis=0))/(train_images_augmented[:,:,:,1].max(axis=0) -train_images_augmented[:,:,:,1].min(axis=0)  )
train_images_augmented[:,:,:,2] = (train_images_augmented[:,:,:,2]-train_images_augmented[:,:,:,2].min(axis=0))/(train_images_augmented[:,:,:,2].max(axis=0) -train_images_augmented[:,:,:,2].min(axis=0)  )
train_images = np.concatenate((train_images,train_images_augmented),axis=0)
train_labels = np.concatenate((train_labels,train_labels),axis=0)
train_images,train_labels = shuffle(train_images,train_labels)

# test_images[:,:,:,0] = (test_images[:,:,:,0]-test_images[:,:,:,0].min(axis=0))/(test_images[:,:,:,0].max(axis=0) -test_images[:,:,:,0].min(axis=0)  )
# test_images[:,:,:,1] = (test_images[:,:,:,1]-test_images[:,:,:,1].min(axis=0))/(test_images[:,:,:,1].max(axis=0) -test_images[:,:,:,1].min(axis=0)  )
# test_images[:,:,:,2] = (test_images[:,:,:,2]-test_images[:,:,:,2].min(axis=0))/(test_images[:,:,:,2].max(axis=0) -test_images[:,:,:,2].min(axis=0)  )

# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

# === Hyper ===
num_epoch =  100
batch_size = 100
print_size = 1
shuffle_size = 2
divide_size = 4

# proportion_rate = 0.01
decay_rate = 0.8
# decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

# learning_rate = 0.00001
# momentum_rate = 0.9















# =========== Layer Class ===========
# === Convolutional Layer ===
class CNNLayer():
      
  def __init__(self,kernel,in_c,out_c,in_c2,out_c2,act,d_act):
    with tf.device('/cpu:0'):
      self.w = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c],stddev=0.05,mean=0.0))
      
      self.act,self.d_act = act,d_act
      self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))
  def getw(self): return [self.w]
  def reg(self): return tf.nn.l2_loss(self.w)

  def feedforward(self,input,strides=1,more=False,res=None):
    self.input = input
    self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,strides,strides,1],padding='SAME')
    self.layerA = self.act(self.layer)

    if more:
      self.rec = tf.nn.avg_pool(res, [1,2,2,1], [1,2,2,1], padding='SAME')
      self.rec = tf.concat((self.rec,self.rec),axis=3)
      self.layerA = self.layerA  + self.rec
    return self.layerA 

  def feedforward_res(self,input,res):
    self.input = input
    self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME')
    self.layerA = self.act(self.layer) + res
    return self.layerA

  # def feedforward_dropout(self,input,droprate):
  #   self.input = input
  #   self.layer = tf.nn.dropout(tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME'),droprate)
  #   self.layerA = self.act(self.layer)
  #   return self.layerA

  # def feedforward_avg(self,input,droprate):
  #   self.input = input
  #   self.layer =  tf.nn.dropout(tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME'),droprate)
  #   self.layerA = self.act(self.layer)
  #   self.layerMean = tf.nn.avg_pool(self.layerA, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  #   return self.layerMean 

  # def backprop(self,gradient):
  #       return 2










# === Make Layers ===
l0_1 = CNNLayer(7,3,32,3,32,  tf_elu,d_tf_elu)
l0_2 = CNNLayer(1,32,32,32,32,tf_elu,d_tf_elu)
l0_3 = CNNLayer(5,32,32,32,32,tf_elu,d_tf_elu)
l0_4 = CNNLayer(1,32,64,32,64,tf_elu,d_tf_elu)

l1_1 = CNNLayer(3,64,64,64,64,tf_elu,d_tf_elu)
l1_2 = CNNLayer(1,64,64,64,64,tf_elu,d_tf_elu)
l1_3 = CNNLayer(2,64,64,64,64,tf_elu,d_tf_elu)
l1_4 = CNNLayer(1,64,128,64,128,tf_elu,d_tf_elu)

l2_1 = CNNLayer(3,128,128,128,128,tf_elu,d_tf_elu)
l2_2 = CNNLayer(1,128,128,128,128,tf_elu,d_tf_elu)
l2_3 = CNNLayer(2,128,128,128,128,tf_elu,d_tf_elu)
l2_4 = CNNLayer(1,128,256,128,256,tf_elu,d_tf_elu)

l3_1 = CNNLayer(3,256,256,256,256,tf_elu,d_tf_elu)
l3_2 = CNNLayer(1,256,256,256,256,tf_elu,d_tf_elu)
l3_3 = CNNLayer(2,256,256,256,256,tf_elu,d_tf_elu)
l3_4 = CNNLayer(1,256,256,256,10,tf_elu,d_tf_elu)

l4_1 = CNNLayer(2,256,256,256,256,tf_elu,d_tf_elu)
l4_2 = CNNLayer(1,256,256,256,256,tf_elu,d_tf_elu)
l4_3 = CNNLayer(1,256,256,256,256,tf_elu,d_tf_elu)
l4_4 = CNNLayer(1,256,10,256,10,tf_elu,d_tf_elu)







# === Make Graph ===
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)
learning_rate  = tf.placeholder(shape=[],dtype=tf.float32)
momentum_rate  = tf.placeholder(shape=[],dtype=tf.float32)


droprate1 = tf.placeholder(shape=[],dtype=tf.float32)
droprate2 = tf.placeholder(shape=[],dtype=tf.float32)
droprate3 = tf.placeholder(shape=[],dtype=tf.float32)
droprate4 = tf.placeholder(shape=[],dtype=tf.float32)

proportion_rate  = tf.placeholder(shape=[],dtype=tf.float32)
iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

layer0_1 = l0_1.feedforward(x)
layer0_2 = l0_2.feedforward_res(layer0_1,res=layer0_1)
layer0_3 = l0_3.feedforward_res(layer0_2,res=layer0_2+decay_propotoin_rate*(layer0_1))
layer0_4 = l0_4.feedforward(layer0_3,2,True,res=layer0_3+decay_propotoin_rate*(layer0_1+layer0_2))

layer1_1 = l1_1.feedforward(layer0_4)
layer1_2 = l1_2.feedforward_res(layer1_1,res=layer1_1 )
layer1_3 = l1_3.feedforward_res(layer1_2,res=layer1_2+decay_propotoin_rate*(layer1_2))
layer1_4 = l1_4.feedforward(layer1_3,2,True,res=layer1_3+decay_propotoin_rate*(layer1_2+layer1_3)) 

layer2_1 = l2_1.feedforward(layer1_4)
layer2_2 = l2_2.feedforward_res(layer2_1,res=layer2_1)
layer2_3 = l2_3.feedforward_res(layer2_2,res=layer2_2+decay_propotoin_rate*(layer2_2))
layer2_4 = l2_4.feedforward(layer2_3,2,True,res=layer2_3+decay_propotoin_rate*(layer2_2+layer2_3))

layer3_1 = l3_1.feedforward(layer2_4)
layer3_2 = l3_2.feedforward_res(layer3_1,res=layer3_1)
layer3_3 = l3_3.feedforward_res(layer3_2,res=layer3_2+decay_propotoin_rate*(layer3_2))
layer3_4 = l3_4.feedforward(layer3_3,2,res=layer3_3+decay_propotoin_rate*(layer3_2+layer3_3))

layer4_1 = l4_1.feedforward(layer3_4)
layer4_2 = l4_2.feedforward_res(layer4_1,res=layer4_1)
layer4_3 = l4_3.feedforward_res(layer4_2,res=layer4_2+decay_propotoin_rate*(layer4_2))
layer4_4 = l4_4.feedforward(layer4_3,2,res=layer4_3+decay_propotoin_rate*(layer4_2+layer4_3))

final_soft = tf_softmax(tf.reshape(layer4_4,[batch_size,-1]))
cost = tf.reduce_sum(-1.0 * (y*tf.log(final_soft) + (1.0-y)*tf.log(1.0-final_soft)))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


regularizer = l0_1.reg() + l0_2.reg()+ l0_3.reg()+ l0_4.reg() +\
              l2_1.reg() + l2_2.reg()+ l2_3.reg()+ l2_4.reg() +\
              l3_1.reg() + l3_2.reg()+ l3_3.reg()+ l3_4.reg() +\
              l4_1.reg() + l4_2.reg()+ l4_3.reg()+ l4_4.reg() 


# --- auto train ---
auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost+0.5*regularizer)
auto_train2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost+0.5*regularizer)





























# === Start the Session ===
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
gpu_options.allow_growth=True
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

            if iter  >= 30:
              feed_dict ={x:current_batch,y:current_batch_label,iter_variable_dil:iter,droprate1:1.0,droprate2:1.0,droprate3:1.0,droprate4:1.0,learning_rate:0.000001,momentum_rate:0.9,proportion_rate:0.6}
              sess_results = sess.run([cost,accuracy,correct_prediction,auto_train],feed_dict=feed_dict)
            elif iter >= 17:
              feed_dict ={x:current_batch,y:current_batch_label,iter_variable_dil:iter,droprate1:1.0,droprate2:1.0,droprate3:1.0,droprate4:1.0,learning_rate:0.000005,momentum_rate:0.9,proportion_rate:1}
              sess_results = sess.run([cost,accuracy,correct_prediction,auto_train],feed_dict=feed_dict)      
            else:
              feed_dict ={x:current_batch,y:current_batch_label,iter_variable_dil:iter,droprate1:1.0,droprate2:1.0,droprate3:1.0,droprate4:1.0,learning_rate:0.00001,momentum_rate:0.9,proportion_rate:0.01}
              sess_results = sess.run([cost,accuracy,correct_prediction,auto_train],feed_dict=feed_dict)

            print("current iter:", iter,' Current batach : ',current_batch_index," current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
          current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
          current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]

          if iter  >= 30:
            feed_dict ={x:current_batch,y:current_batch_label,iter_variable_dil:iter,droprate1:1.0,droprate2:1.0,droprate3:1.0,droprate4:1.0,proportion_rate:0.6}
            sess_results = sess.run([cost,accuracy,correct_prediction],feed_dict=feed_dict)
          elif iter >= 17:
            feed_dict ={x:current_batch,y:current_batch_label,iter_variable_dil:iter,droprate1:1.0,droprate2:1.0,droprate3:1.0,droprate4:1.0,proportion_rate:1}
            sess_results = sess.run([cost,accuracy,correct_prediction],feed_dict=feed_dict)      
          else:
            feed_dict ={x:current_batch,y:current_batch_label,iter_variable_dil:iter,droprate1:1.0,droprate2:1.0,droprate3:1.0,droprate4:1.0,proportion_rate:0.01}
            sess_results = sess.run([cost,accuracy,correct_prediction],feed_dict=feed_dict)

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
  plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r',label="Train")
  plt.plot(range(len(train_cost_overtime)),test_cost_overtime,color='b',label='Test')
  plt.legend()
  plt.title('Cost over time')
  plt.show()

  plt.figure()
  plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='r',label="Train")
  plt.plot(range(len(train_acc_overtime)),test_acc_overtime,color='b',label='Test')
  plt.legend()
  plt.title('Acc over time')
  plt.show()
        








# -- end code --