import tensorflow as tf
import numpy as np
from numpy import float32

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return 7 

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf.log(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tansh(x): return 1.0 - tf.square(tf_tanh(x))


# Make each class for the networks
class prepnetwork():
    
    def __init__(self):
        self.w1 = tf.Variable(tf.random_normal([3,3,3,50]))
        self.w2 = tf.Variable(tf.random_normal([3,3,50,50]))
        self.w3 = tf.Variable(tf.random_normal([3,3,50,50]))
        self.w4 = tf.Variable(tf.random_normal([3,3,50,50]))
        self.w5 = tf.Variable(tf.random_normal([3,3,50,3]))

    def feedforward(self,input=None):
        layer1 = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        layer2 = tf.nn.conv2d(layer1,self.w2,strides=[1,1,1,1],padding='SAME')
        layer3 = tf.nn.conv2d(layer2,self.w3,strides=[1,1,1,1],padding='SAME')
        layer4 = tf.nn.conv2d(layer3,self.w4,strides=[1,1,1,1],padding='SAME')
        layer5 = tf.nn.conv2d(layer4,self.w5,strides=[1,1,1,1],padding='SAME')
        return layer5

class hididingnetwork():
    
    def __init__(self):
        self.w1 = tf.Variable(tf.random_normal([4,4,6,50]))
        self.w2 = tf.Variable(tf.random_normal([4,4,50,50]))
        self.w3 = tf.Variable(tf.random_normal([4,4,50,50]))
        self.w4 = tf.Variable(tf.random_normal([4,4,50,50]))
        self.w5 = tf.Variable(tf.random_normal([4,4,50,3]))

    def feedforward(self,input=None):
        layer1 = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        layer2 = tf.nn.conv2d(layer1,self.w2,strides=[1,1,1,1],padding='SAME')
        layer3 = tf.nn.conv2d(layer2,self.w3,strides=[1,1,1,1],padding='SAME')
        layer4 = tf.nn.conv2d(layer3,self.w4,strides=[1,1,1,1],padding='SAME')
        layer5 = tf.nn.conv2d(layer4,self.w5,strides=[1,1,1,1],padding='SAME')
        return layer5

class revealnetwork():
    
    def __init__(self):
        self.w1 = tf.Variable(tf.random_normal([5,5,3,50]))
        self.w2 = tf.Variable(tf.random_normal([5,5,50,50]))
        self.w3 = tf.Variable(tf.random_normal([5,5,50,50]))
        self.w4 = tf.Variable(tf.random_normal([5,5,50,50]))
        self.w5 = tf.Variable(tf.random_normal([5,5,50,3]))

    def feedforward(self,input=None):
        layer1 = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        layer2 = tf.nn.conv2d(layer1,self.w2,strides=[1,1,1,1],padding='SAME')
        layer3 = tf.nn.conv2d(layer2,self.w3,strides=[1,1,1,1],padding='SAME')
        layer4 = tf.nn.conv2d(layer3,self.w4,strides=[1,1,1,1],padding='SAME')
        layer5 = tf.nn.conv2d(layer4,self.w5,strides=[1,1,1,1],padding='SAME')
        return layer5

# Declare the Objects and the networks
prepnetwork = prepnetwork()
hididingnetwork = hididingnetwork()
revealnetwork = revealnetwork()

# Make the Graph
s = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
c = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)

# --- Prep part ---
layer_prep  = prepnetwork.feedforward(s)

# --- Cover part ---
cover_and_secret = tf.concat([layer_prep,c],3)
layer_hiddne = hididingnetwork.feedforward(cover_and_secret) 

# --- Reveal part ---
layer_reveal = revealnetwork.feedforward(layer_hiddne) 


# Start the training Session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    temp = float32(np.ones((1,32,32,3)))

    sess_results = sess.run(layer_reveal,feed_dict={s:temp,c:temp})

    print(sess_results.shape)







# -- end code --