import os
import numpy
from matplotlib import pyplot as plt, cm
import io,gzip,zipfile
import numpy as np,sys,os
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

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
  maks2 = tf_elu(tf.cast(tf.less_equal(x,0),dtype=tf.float32) * x)
  return mask1 + mask2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

# --- get data ---
PathDicom = "./data/cifar-10-batches-py/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if not ".html" in filename.lower() and not  ".meta" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

batch0 = unpickle(lstFilesDCM[0])
batch1 = unpickle(lstFilesDCM[1])
batch2 = unpickle(lstFilesDCM[2])
batch3 = unpickle(lstFilesDCM[3])
batch4 = unpickle(lstFilesDCM[4])
onehot_encoder = OneHotEncoder(sparse=True)

train_batch = np.vstack((batch0[b'data'],batch1[b'data'],batch2[b'data'],batch3[b'data'],batch4[b'data']))
train_label = np.expand_dims(np.hstack((batch0[b'labels'],batch1[b'labels'],batch2[b'labels'],batch3[b'labels'],batch4[b'labels'])).T,axis=1).astype(np.float32)
train_labels = onehot_encoder.fit_transform(train_label).toarray().astype(np.float32)

test_batch = unpickle(lstFilesDCM[5])[b'data']
test_label = np.expand_dims(np.array(unpickle(lstFilesDCM[5])[b'labels']),axis=0).T.astype(np.float32)
test_labels = onehot_encoder.fit_transform(test_label).toarray().astype(np.float32)

# reshape data
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))

# rotate data
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)

# Normalize data from 0 to 1
train_batch[:,:,:,0] = (train_batch[:,:,:,0] - train_batch[:,:,:,0].min(axis=0))/(train_batch[:,:,:,0].max(axis=0)-train_batch[:,:,:,0].min(axis=0))
train_batch[:,:,:,1] = (train_batch[:,:,:,1] - train_batch[:,:,:,1].min(axis=0))/(train_batch[:,:,:,1].max(axis=0)-train_batch[:,:,:,1].min(axis=0))
train_batch[:,:,:,2] = (train_batch[:,:,:,2] - train_batch[:,:,:,2].min(axis=0))/(train_batch[:,:,:,2].max(axis=0)-train_batch[:,:,:,2].min(axis=0))

test_batch[:,:,:,0] = (test_batch[:,:,:,0] - test_batch[:,:,:,0].min(axis=0))/(test_batch[:,:,:,0].max(axis=0)-test_batch[:,:,:,0].min(axis=0))
test_batch[:,:,:,1] = (test_batch[:,:,:,1] - test_batch[:,:,:,1].min(axis=0))/(test_batch[:,:,:,1].max(axis=0)-test_batch[:,:,:,1].min(axis=0))
test_batch[:,:,:,2] = (test_batch[:,:,:,2] - test_batch[:,:,:,2].min(axis=0))/(test_batch[:,:,:,2].max(axis=0)-test_batch[:,:,:,2].min(axis=0))

train_images = train_batch
test_images  = test_batch

# === Hyper Parameter ===
num_epoch =  100
batch_size = 100
print_size = 1
shuffle_size = 1
divide_size = 5

beta1,beta2 = 0.9,0.999
adam_e = 0.00000001

proportion_rate = 1000
decay_rate = 0.08

one_channel = 4

# === make classes ====
l1_1 = ConLayer(3,3,one_channel,tf_ReLU,d_tf_ReLu)
l1_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l1_s = ConLayer(1,3,one_channel,tf_ReLU,d_tf_ReLu)

l2_1_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_1_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_1_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l2_2_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_2_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_2_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l2_3_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_3_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l2_3_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l3_1_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_1_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_1_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l3_2_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_2_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_2_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)

l3_3_1 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_3_2 = ConLayer(3,one_channel,one_channel,tf_ReLU,d_tf_ReLu)
l3_3_s = ConLayer(1,one_channel,one_channel,tf_ReLU,d_tf_ReLu)



# --- make graph ----
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

layer1_1 = l1_1.feedforward(x)
layer1_2 = l1_2.feedforward(layer1_1)
layer1_s = l1_s.feedforward(x)
layer1_add = layer1_s + layer1_2

# --- node layer 2 -----
layer2_1_1 = l2_1_1.feedforward(layer1_add)
layer2_1_2 = l2_1_2.feedforward(layer2_1_1)
layer2_1_s = l2_1_s.feedforward(layer1_add)
layer2_1_add = layer2_1_s + layer2_1_2

layer2_2_1 = l2_2_1.feedforward(layer1_add)
layer2_2_2 = l2_2_2.feedforward(layer2_1_1)
layer2_2_s = l2_2_s.feedforward(layer1_add)
layer2_2_add = layer2_2_s + layer2_2_2

layer2_3_1 = l2_3_1.feedforward(layer1_add)
layer2_3_2 = l2_3_2.feedforward(layer2_3_1)
layer2_3_s = l2_3_s.feedforward(layer1_add)
layer2_3_add = layer2_3_s + layer2_3_2

# --- node layer 3 -----
layer3_Input = layer2_1_add + layer2_2_add + layer2_3_add
layer3_1_1 = l3_1_1.feedforward(layer3_Input)
layer3_1_2 = l3_1_2.feedforward(layer3_1_1)
layer3_1_s = l3_1_s.feedforward(layer3_Input)
layer3_1_add = layer3_1_s + layer3_1_2

layer3_2_1 = l3_2_1.feedforward(layer3_Input)
layer3_2_2 = l3_2_2.feedforward(layer3_2_1)
layer3_2_s = l3_2_s.feedforward(layer3_Input)
layer3_2_add = layer3_2_s + layer3_2_2

layer3_3_1 = l3_3_1.feedforward(layer3_Input)
layer3_3_2 = l3_3_2.feedforward(layer3_3_1)
layer3_3_s = l3_3_s.feedforward(layer3_Input)
layer3_3_add = layer3_3_s + layer3_3_2

# ---- fully connected layer ----
layer4_Input = tf.reshape(tf.concat([layer3_1_add,layer3_2_add,layer3_3_add],axis=3),[batch_size,-1])
print(layer4_Input.shape)




# -- end code ---