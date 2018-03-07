import tensorflow as tf
import numpy as np
from numpy import float32

def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return 7 

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf.log(x))




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

class hidiingnetwork():
    
    def __init__(self):
        self.w1 = tf.Variable(tf.random_normal([4,4,3,50]))
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

prepnetwork = prepnetwork()


temp = float32(np.ones((1,64,64,3)))
print(temp.shape)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    dddd = prepnetwork.feedforward(temp).eval()

    print(dddd.shape)







# -- end code --