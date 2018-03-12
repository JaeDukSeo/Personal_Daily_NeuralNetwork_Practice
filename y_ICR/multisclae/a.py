import tensorflow as tf
import numpy as np

np.random.seed(678)
tf.set_random_seed(6786)

def tf_Relu(x): return tf.nn.relu(x)
def d_tf_ReLu(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)

class contextLayer():
    
    def __init__(self,kernelsize,inchannel,outchannel):
        
        self.w = tf.Variable(tf.random_normal([kernelsize,kernelsize,inchannel,outchannel]))

    def feedforward(self,input=None,dilationfactor=None,Same=False):
        
        if Same:
            self.layer  = tf.nn.conv2d(input,self.w, strides=[1,1,1,1],padding="VALID")
        else:
            self.layer  = tf.nn.atrous_conv2d(input,self.w, rate=dilationfactor,padding="VALID")
            
            
        self.layerA = tf_Relu(self.layer)
        return self.layerA

    def backprop(self,gradient=None,dilation_factor = None):
        return 3


layer1 = contextLayer(3,1,1)
layer2 = contextLayer(3,1,1)
layer3 = contextLayer(3,1,1)
layer4 = contextLayer(3,1,1)

layer5 = contextLayer(3,1,1)
layer6 = contextLayer(3,1,1)
layer7 = contextLayer(3,1,1)
layer8 = contextLayer(3,1,1)
layer9 = contextLayer(2,1,1)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    temp = np.ones((1,64,64,1)).astype(np.float32)

    l1 = layer1.feedforward(temp,1).eval()
    l2 = layer2.feedforward(l1,1).eval()
    l3 = layer3.feedforward(l2,1).eval()
    l4 = layer4.feedforward(l3,2).eval()

    l5 = layer5.feedforward(l4,4).eval()
    l6 = layer6.feedforward(l5,8).eval()
    l7 = layer7.feedforward(l6,11).eval()
    l8 = layer8.feedforward(l7,1).eval()

    print(l8.shape)

    l9 = layer9.feedforward(l8,True).eval()
    
    print(l9.shape)
    print(np.reshape(l9,(1,-1)).shape )




# -- end code --