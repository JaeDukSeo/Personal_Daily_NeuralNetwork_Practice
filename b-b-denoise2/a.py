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
one = np.zeros((119,512,512))
two = np.zeros((119,512,512))
three = np.zeros((119,512,512))

lung_data1 = zipfile.ZipFile("./data/lung_data_1.zip", 'r')
lung_data2 = zipfile.ZipFile("./data/lung_data_2.zip", 'r')
lung_data3 = zipfile.ZipFile("./data/lung_data_3.zip", 'r')

lung_data1_list = lung_data1.namelist()
lung_data2_list = lung_data2.namelist()
lung_data3_list = lung_data3.namelist()

for current_file_index in range(0,len(lung_data1.namelist()) - 9 ):
  one[current_file_index,:,:] =   imread(io.BytesIO(lung_data1.open(lung_data1_list[current_file_index]).read()),mode='F')
  two[current_file_index,:,:] =   imread(io.BytesIO(lung_data2.open(lung_data2_list[current_file_index]).read()),mode='F')
  three[current_file_index,:,:] = imread(io.BytesIO(lung_data3.open(lung_data3_list[current_file_index]).read()),mode='F')

# --- normalize data batch normalization ----
one = (one-one.min()) / (one.max()-one.min()) 
two = (two-two.min()) / (two.max()-two.min()) 
three = (three-three.min()) / (three.max()-three.min()) 

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

    grad_middle = tf.nn.multiply(grad_part1,grad_part2)

    grad_w = tf.nn.conv2d_backprop_filter(
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


    return grad_pass,update_w

# --- hyper parameter ---
num_epoch = 300
batch_size = 10

beta1,beta2 = 0.9,0.999
adam_e = 1e-8

one_channel = 128

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




  


# -- end code --



