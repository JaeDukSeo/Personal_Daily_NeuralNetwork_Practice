import tensorflow as tf
import numpy as np,sys
from numpy import float32
import matplotlib.pyplot as plt

np.random.seed(678)
tf.set_random_seed(678)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return 7 

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf.log(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

def tf_softmax(x): return tf.nn.softmax(x)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    X = np.asarray(dict[b'data'].T).astype("uint8")
    Yraw = np.asarray(dict[b'labels'])
    Y = np.zeros((10,10000))
    for i in range(10000):
        Y[Yraw[i],i] = 1
    names = np.asarray(dict[b'filenames'])
    return X,Y,names


class RCNN():
    
    def __init__(self,time_seq=None,Height=None,width=None,
                input_dim=None, hidden_dim =None,act=None,d_act=None):
        
        self.w_x = tf.Variable(tf.random_normal([3,3,input_dim,hidden_dim]))
        self.w_h = tf.Variable(tf.random_normal([3,3,hidden_dim,hidden_dim]))

        self.hidden  = tf.Variable(tf.zeros([time_seq,Height,width,hidden_dim]))
        self.hiddenA = tf.Variable(tf.zeros([time_seq,Height,width,hidden_dim]))

        self.act,self.d_act = act,d_act

    def feedforward(self,input=None,timestamp=None):
        
        self.layer  = tf.nn.conv2d(input,self.w_x,strides=[1,1,1,1],padding='SAME')  + \
                      tf.nn.conv2d(
                          tf.expand_dims(self.hidden[timestamp,:,:,:],axis=0),self.w_h,strides=[1,1,1,1],padding='SAME') 
        self.layerA = self.act(self.layer)

        assign_hidden = []
        assign_hidden.append(tf.assign(self.hidden[timestamp+1,:,:,:],self.layer))
        assign_hidden.append(tf.assign(self.hiddenA[timestamp+1,:,:,:],self.layerA))

        return self.layerA,assign_hidden

class FNN():
    
    def __init__(self,input_shape=None,output_shape=None):
        
        self.w = tf.Variable(tf.random_normal(shape=[input_shape,output_shape]))

# ------- Preprocess Data --------
X,Y,names = unpickle('../../z_CIFAR_data/cifar10batchespy/data_batch_1')
Y = Y.T
X = np.reshape(X,(3,32,32,10000)).transpose([3,1,2,0])
X = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

# --- Hyper Parameter ----
batch_size = 100

# ---- Make Objects ----
l1_RCNN = RCNN(4,32,32,1,5,tf_tanh,d_tf_tanh)
l2_RCNN = RCNN(4,32,32,5,7,tf_tanh,d_tf_tanh)
l3_RCNN = RCNN(4,32,32,7,1,tf_tanh,d_tf_tanh)

# ---- Make Graph -----
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)
time_stamp = tf.constant(0)

l1_1,l1_1w = l1_RCNN.feedforward(tf.expand_dims(x[:,:,:,0],axis=3),time_stamp+0)
l1_2,l1_2w = l1_RCNN.feedforward(tf.expand_dims(x[:,:,:,1],axis=3),time_stamp+1)
l1_3,l1_3w = l1_RCNN.feedforward(tf.expand_dims(x[:,:,:,2],axis=3),time_stamp+2)

l2_1,l2_1w = l2_RCNN.feedforward(l1_1,time_stamp+0)
l2_2,l2_2w = l2_RCNN.feedforward(l1_2,time_stamp+1)
l2_3,l2_3w = l2_RCNN.feedforward(l1_3,time_stamp+2)

l3_1,l3_1w = l3_RCNN.feedforward(l2_1,time_stamp+0)
l3_2,l3_2w = l3_RCNN.feedforward(l2_2,time_stamp+1)
l3_3,l3_3w = l3_RCNN.feedforward(l2_3,time_stamp+2)

print(tf.reshape(l3_3,[batch_size,-1]).shape)
print(l3_3.shape)

final_softmax = tf_softmax(l3_3)
cost_dil = tf.reduce_sum( -1 * ( y*tf.log(final_softmax) + (1-y) * tf.log(1-final_softmax ) ) )
correct_prediction = tf.equal(tf.argmax(final_softmax, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# --- train sesion ----
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

