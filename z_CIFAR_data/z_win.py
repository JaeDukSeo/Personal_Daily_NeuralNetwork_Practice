import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

np.random.seed(6789)
tf.set_random_seed(678)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_acrtan(x): return tf.atan(x)
def d_tf_arctan(x): return 1/(1+tf.square(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

# make class
class CNNLayer():
    
    def __init__(self,kernel,in_c,out_c,padding,act,d_act):
        self.w = tf.Variable(tf.random_normal([kernel,kernel,in_c,out_c]))
        self.act,self.d_act = act,d_act
        
    def feedforward(self,input):
        self.input  = input 
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding=padding)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient):
        return 3

class FNNLayer():
    
    def __init__(self,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.random_normal([in_c,out_c]))
        self.act,self.d_act = act,d_act

    def feedforward(self,input,resinput):
        self.input  = input 
        self.layer  = tf.matmul(input,self.w)
        self.layerA = self.act(self.layer)
        return self.layerA        
        

# Get the Train data
data = unpickle('cifar100python/train')
train_batch = data[b'data']
train_label = data[b'fine_labels']

# Get the Test Data
data = unpickle('cifar100python/test')
test_batch = data[b'data']
test_label = data[b'fine_labels']

# Normalize
train_batch = (train_batch - train_batch.min(axis=0))/(train_batch.max(axis=0)-train_batch.min(axis=0))
test_batch = (test_batch - test_batch.min(axis=0))/(test_batch.max(axis=0)-test_batch.min(axis=0))

# reshape data
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))

# rotate data
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)



# -- end code --