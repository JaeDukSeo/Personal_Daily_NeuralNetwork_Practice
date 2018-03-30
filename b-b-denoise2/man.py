import os
import numpy
from matplotlib import pyplot as plt, cm
import io
import gzip
import zipfile
import numpy as np,sys,os
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show,ion
from scipy.ndimage import imread
from sklearn.utils import shuffle
import tensorflow as tf

tf.set_random_seed(789)
np.random.seed(568)

# -1 Tf activation functions
def tf_arctan(x): return tf.atan(x)
def d_tf_arctan(x): return 1.0/(1+tf.square(x))

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLu(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)

def tf_tanh(x):return tf.tanh(x)
def d_tf_tanh(x):return 1. - tf.square(tf_tanh(x))

def tf_log(x):return tf.sigmoid(x)
def d_tf_log(x):return tf.sigmoid(x) * (1.0 - tf.sigmoid(x))

def tf_elu(x): return tf.elu(x)
def d_tf_elu(x): 
  mask1 = tf.cast(tf.greater(x,0),dtype=tf.flaot32)
  maks2 = tf_elu(tf.less_equal(x,0))
  return mask1 + mask2

# ---- prepare data ----
one = np.zeros((110,512,512,1))
two = np.zeros((110,512,512,1))
three = np.zeros((110,512,512,1))

lung_data1 = zipfile.ZipFile("./data/lung_data_1.zip", 'r')
lung_data2 = zipfile.ZipFile("./data/lung_data_2.zip", 'r')
lung_data3 = zipfile.ZipFile("./data/lung_data_3.zip", 'r')

lung_data1_list = lung_data1.namelist()
lung_data2_list = lung_data2.namelist()
lung_data3_list = lung_data3.namelist()

for current_file_index in range(0,len(lung_data1.namelist()) - 9 ):
  one[current_file_index,:,:] =   np.expand_dims(imread(io.BytesIO(lung_data1.open(lung_data1_list[current_file_index]).read()),mode='F').astype(np.float32),axis=3)
  two[current_file_index,:,:] =    np.expand_dims(imread(io.BytesIO(lung_data2.open(lung_data2_list[current_file_index]).read()),mode='F').astype(np.float32),axis=3)
  three[current_file_index,:,:] = np.expand_dims( imread(io.BytesIO(lung_data3.open(lung_data3_list[current_file_index]).read()),mode='F').astype(np.float32),axis=3)

# --- normalize data batch normalization ----
one = (one-one.min()) / (one.max()-one.min()) 
two = (two-two.min()) / (two.max()-two.min()) 
three = (three-three.min()) / (three.max()-three.min()) 

# split train and test
train_images  = np.vstack((one,two))
test_images   = three









# ---- make class ----
class ConLayer():
  
  def __init__(self,kernel,in_c,out_c,act,d_act):
    self.w  = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c], stddev=0.05))
    self.act,self.d_act = act,d_act
    self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

  def feedforward(self,input):
    self.input = input
    self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='SAME')
    self.layerA = self.act(self.layer)
    return self.layerA

  def backprop(self,gradient):
    grad_part1 = gradient
    grad_part2 = self.d_act(self.layer)
    grad_part3 = self.input

    grad_middle = tf.multiply(grad_part1,grad_part2)

    grad = tf.nn.conv2d_backprop_filter(
      input = grad_part3,
      filter_sizes = self.w.shape,
      out_backprop = grad_middle,
      strides=[1,1,1,1],padding='SAME'
    )

    grad_pass = tf.nn.conv2d_backprop_input(
      input_sizes = [batch_size] + list(self.input.shape[1:]),
      filter = self.w,
      out_backprop = grad_middle,
      strides=[1,1,1,1],padding='SAME'
    )

    update_w = []

    update_w.append( tf.assign( self.m,self.m * beta1 + (1-beta1) * grad     )   )
    update_w.append( tf.assign( self.v,self.v * beta2 + (1-beta2) * grad  ** 2   )   )

    m_hat = self.m/(1-beta1)
    v_hat = self.v/(1-beta2)
    adam_middle = init_lr / ( tf.sqrt(v_hat) + adam_e)

    update_w.append(
      tf.assign(self.w, self.w - adam_middle * m_hat   )
    )

    return grad_pass,update_w

# --- hyper parameter ---
num_epoch = 100
batch_size = 10

init_lr = 0.0001

beta1,beta2 = 0.9,0.999
adam_e = 1e-8

one_channel = 8
print_size = 1

proportion_rate = 1000
decay_rate = 0.08







# --- make layers ---
l1_1 = ConLayer(3,1,one_channel,tf_ReLU,d_tf_ReLu)
l1_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l1_s = ConLayer(1,1,one_channel,tf_ReLU,d_tf_ReLu)

l2_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l3_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_2 = ConLayer(3,one_channel,1,tf_ReLU,d_tf_ReLu)
l3_s = ConLayer(1,one_channel,1,tf_ReLU,d_tf_ReLu)














# --- make graph ---
x = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)

iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

layer1_1 = l1_1.feedforward(x)
layer1_2 = l1_2.feedforward(layer1_1)
layer1_s = l1_s.feedforward(x)
layer1_add = layer1_s + layer1_2

layer2_1 = l2_1.feedforward(layer1_add)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_s = l2_s.feedforward(layer1_add)
layer2_add = layer2_s + layer2_2

layer3_1 = l3_1.feedforward(layer2_add)
layer3_2 = l3_2.feedforward(layer3_1)
layer3_s = l3_s.feedforward(layer2_add)
layer3_add = layer3_s + layer3_2

cost = tf.reduce_mean(tf.square(layer3_add-y)*0.5)

# # --- man back prop ---
grad3_s,grad3_sw = l3_s.backprop(layer3_add-y)
grad3_2,grad3_2w = l3_2.backprop(layer3_add-y)
grad3_1,grad3_1w = l3_1.backprop(grad3_2)

grad2_s,grad2_sw = l2_s.backprop(grad3_1+grad3_s+decay_propotoin_rate * ( (layer3_add-y) +grad3_2  ) ) 
grad2_2,grad2_2w = l2_2.backprop(grad3_1+grad3_s+decay_propotoin_rate * ( (layer3_add-y) +grad3_2  ))
grad2_1,grad2_1w = l2_1.backprop(grad2_2+decay_propotoin_rate * ( (layer3_add-y) +grad3_2 + grad3_1 ) ) 

grad1_s,grad1_sw = l1_s.backprop(grad2_1+grad2_s+decay_propotoin_rate * ( (layer3_add-y) +grad3_2 + grad3_1+grad2_2 ))
grad1_2,grad1_2w = l1_2.backprop(grad2_1+grad2_s+decay_propotoin_rate * ( (layer3_add-y) +grad3_2 + grad3_1+grad2_2 ))
grad1_1,grad1_1w = l1_1.backprop(grad1_2+decay_propotoin_rate * ( (layer3_add-y) +grad3_2 + grad3_1+grad2_2+grad1_s ))

grad_update = grad3_sw + grad3_2w + grad3_1w + \
              grad2_sw + grad2_2w + grad2_1w + \
              grad1_sw + grad1_2w + grad1_1w 
















# --- make session ---
with tf.Session() as sess: 

  sess.run(tf.global_variables_initializer())

  train_total_cost =0
  train_cost_overtime = []

  test_total_cost = 0
  test_cost_overtime = []

  # epcoh
  for iter in range(num_epoch):
        
    train_images = shuffle(train_images)

    for current_batch_index in range(0,len(train_images),batch_size):
      current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
      current_batch_noise =  current_batch * 0.5 * np.random.uniform(0,5,size=(current_batch.shape[0],current_batch.shape[1],current_batch.shape[2],current_batch.shape[3])) 
      sess_results = sess.run([cost,grad_update],feed_dict={x:current_batch_noise,y:current_batch,iter_variable_dil:iter})
      print("Iter: ", iter , " Cost %.10f"%sess_results[0],end='\r')
      train_total_cost = train_total_cost + sess_results[0]
    
    print("\n----- testing iter ",iter,' ---------')

    for current_batch_index in range(0,len(test_images),batch_size):
      current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
      current_batch_noise =  current_batch * 0.5 * np.random.uniform(0,5,size=(current_batch.shape[0],current_batch.shape[1],current_batch.shape[2],current_batch.shape[3])) 
      sess_results = sess.run([cost],feed_dict={x:current_batch_noise,y:current_batch})
      print("Iter: ", iter , " Cost %.10f"%sess_results[0],end='\r')
      test_total_cost = test_total_cost + sess_results[0]

    # store
    train_cost_overtime.append(train_total_cost/(len(train_images)/batch_size ) )
    test_cost_overtime.append(test_total_cost/(len(test_images)/batch_size ) )

    if iter%print_size == 0:
      print('------------------------')       
      print("Avg Train Cost: ", train_cost_overtime[-1])
      print("Avg Test Cost: ", test_cost_overtime[-1])
      print('------------------------\n\n')       

    test_total_cost = 0
    train_total_cost = 0

  # see three examples
  current_batch = np.expand_dims(test_images[4,:,:,:],axis=0)
  current_batch_noise =  current_batch * 0.5 * np.random.uniform(0,5,size=(current_batch.shape[0],current_batch.shape[1],current_batch.shape[2],current_batch.shape[3])) 
  sess_results = sess.run(layer3_add,feed_dict={x:current_batch_noise})

  sess_results = (sess_results-sess_results.min()) /  (sess_results.max()- sess_results.min())

  plt.imshow(np.squeeze(current_batch),cmap='gray')
  plt.show()

  plt.imshow(np.squeeze(current_batch_noise),cmap='gray')
  plt.show()

  plt.imshow(np.squeeze(sess_results),cmap='gray')
  plt.show()

  current_batch = np.expand_dims(test_images[100,:,:,:],axis=0)
  current_batch_noise =  current_batch * 0.5 * np.random.uniform(0,5,size=(current_batch.shape[0],current_batch.shape[1],current_batch.shape[2],current_batch.shape[3])) 
  sess_results = sess.run(layer3_add,feed_dict={x:current_batch_noise})

  sess_results = (sess_results-sess_results.min()) /  (sess_results.max()- sess_results.min())

  plt.imshow(np.squeeze(current_batch),cmap='gray')
  plt.show()

  plt.imshow(np.squeeze(current_batch_noise),cmap='gray')
  plt.show()

  plt.imshow(np.squeeze(sess_results),cmap='gray')
  plt.show()  

  current_batch = np.expand_dims(test_images[57,:,:,:],axis=0)
  current_batch_noise =  current_batch * 0.5 * np.random.uniform(0,5,size=(current_batch.shape[0],current_batch.shape[1],current_batch.shape[2],current_batch.shape[3])) 
  sess_results = sess.run(layer3_add,feed_dict={x:current_batch_noise})

  sess_results = (sess_results-sess_results.min()) /  (sess_results.max()- sess_results.min())

  plt.imshow(np.squeeze(current_batch),cmap='gray')
  plt.show()

  plt.imshow(np.squeeze(current_batch_noise),cmap='gray')
  plt.show()

  plt.imshow(np.squeeze(sess_results),cmap='gray')
  plt.show()
  

  


# -- end code --



