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

for current_file_index in range(0,len(lung_data1.namelist())):
  one[current_file_index,:,:] =   imread(io.BytesIO(lung_data1.open(lung_data1_list[current_file_index]).read()),mode='F')
  two[current_file_index,:,:] =   imread(io.BytesIO(lung_data2.open(lung_data2_list[current_file_index]).read()),mode='F')
  three[current_file_index,:,:] = imread(io.BytesIO(lung_data3.open(lung_data3_list[current_file_index]).read()),mode='F')

# --- normalize data batch normalization ----
one = (one-one.min()) / (one.max()-one.min()) 
two = (two-two.min()) / (two.max()-two.min()) 
three = (three-three.min()) / (three.max()-three.min()) 





# -- end code --



