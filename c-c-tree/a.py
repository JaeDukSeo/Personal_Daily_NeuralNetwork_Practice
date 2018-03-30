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

# Normalize data from 0 to 1
train_batch = (train_batch - train_batch.min(axis=0))/(train_batch.max(axis=0)-train_batch.min(axis=0))
test_batch = (test_batch - test_batch.min(axis=0))/(test_batch.max(axis=0)-test_batch.min(axis=0))

# reshape data
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))

# rotate data
train_images = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)
test_images = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)








# -- end code ---