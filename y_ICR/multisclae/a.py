import tensorflow as tf
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle

np.random.seed(678)
tf.set_random_seed(6786)

def tf_Relu(x): return tf.nn.relu(x)
def d_tf_ReLu(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_softmax(x): return tf.nn.softmax(x)
# =============== Create Class =========== 
class contextLayer():
    
    def __init__(self,kernelsize,inchannel,outchannel):
        self.w = tf.Variable(tf.random_normal([kernelsize,kernelsize,inchannel,outchannel]))
    def getw(self): return self.w

    def feedforward(self,input=None,dilationfactor=None,Same=False):
        self.layer  = tf.nn.atrous_conv2d(input,self.w, rate=dilationfactor,padding="SAME")
        self.layerA = tf_Relu(self.layer)
        return self.layerA

    def backprop(self,gradient=None,dilation_factor = None):
        return 3

class FCNN():
    
    def __init__(self,input,output):
        self.w = tf.Variable(tf.random_normal([input,output]))
    def getw(self): return self.w
    def feedforward(self,input):
        self.layer  = tf.matmul(input,self.w)
        self.layerA =  tf_log(self.layer)
        return self.layerA
# =============== Create Class =========== 

# Process Data
mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
testing_images, testing_lables =mnist.test.images,mnist.test.labels
training_images,training_lables =mnist.train.images,mnist.train.labels

testing_images = np.reshape(testing_images,(10000,28,28,1))
training_images = np.reshape(training_images,(55000,28,28,1))

# Hyper Parameters
num_epoch = 100
batch_size = 1000
learning_rate = 0.0001

# Declare Models 
layer1 = contextLayer(3,1,1)
layer2 = contextLayer(3,1,1)
layer3 = contextLayer(3,1,1)
layer4 = contextLayer(3,1,1)

layer5 = contextLayer(3,1,1)
layer6 = contextLayer(3,1,1)
layer7 = contextLayer(3,1,1)
layer8 = contextLayer(3,1,1)

layer9  = FCNN(28*28,1024)
layer10 = FCNN(1024,10)

l1w,l2w,l3w,l4w = layer1.getw(),layer2.getw(),layer3.getw(),layer4.getw()
l5w,l6w,l7w,l8w = layer5.getw(),layer6.getw(),layer7.getw(),layer8.getw()
l9w,l10w = layer9.getw(),layer10.getw()

# Create Graph
x = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

l1 = layer1.feedforward(x,1)
l2 = layer2.feedforward(l1,1)
l3 = layer3.feedforward(l2,1)
l4 = layer4.feedforward(l3,2)

l5 = layer5.feedforward(l4,4)
l6 = layer6.feedforward(l5,8)
l7 = layer7.feedforward(l6,16)
l8 = layer8.feedforward(l7,1)

l9Input = tf.reshape(l8,(batch_size,-1))
l9  = layer9.feedforward(l9Input)
l10 = layer10.feedforward(l9)
final_soft = tf_softmax(l10)

cost = -1.0 * ( y*tf.log(final_soft) + (1-y) * tf.log(1-final_soft))

auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=[l1w,l2w,l3w,l4w,
                                                                                        l5w,l6w,l7w,l8w,
                                                                                        l9w,l10w])

# Create Session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        # every iter shuffle the train set
        training_images,training_lables = shuffle(training_images,training_lables)

        for current_batch_index in range(0,len(training_images),batch_size):
            current_batch = training_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = training_lables[current_batch_index:current_batch_index+batch_size,:]
            sess_results = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_batch_label})
            print("Current Iter: ",iter, " Current Batch Index : ", current_batch_index, " Current cost: ", np.sum(sess_results[0]),end='\r')

        if iter%10==0:
            print('\n')





# -- end code --