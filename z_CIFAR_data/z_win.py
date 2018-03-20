import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

np.random.seed(6789)
tf.set_random_seed(678)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# make class
class CNNLayer():
    
    def __init__(self,kernel,in_c,out_c,act,d_act):
        self.w = tf.Variable(tf.random_normal([kernel,kernel,in_c,out_c]))
        self.act,self.d_act = act,d_act
        
    def feedforward(self,input):
        self.input  = input 
        self.layer  = tf.nn.conv2d(input,self.w,strides=[1,1,1,1],padding='VALID')
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





# -- end code --