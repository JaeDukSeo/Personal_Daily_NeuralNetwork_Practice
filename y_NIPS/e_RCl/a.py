import tensorflow as tf
import numpy as np,sys
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

np.random.seed(678)
tf.set_random_seed(5678)

# activation
def tf_log(x): return tf.Sigmoid(x)
def d_tf_log(x): return tf_log(x) * ( 1.0 - tf_log(x))

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLU(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_arctan(x): return tf.atan(x)
def d_tf_acrtan(x): return 1/(1 + tf.square(x))

def tf_softmax(x): return tf.nn.softmax(x)

# Make Class
class RCNN():
    
    def __init__(self,timestamp,
                x_in,x_out,
                x_kernel,h_kernel,width_height,
                act,d_act):
        
        self.w_x = tf.Variable(tf.random_normal([x_kernel,x_kernel,x_in,x_out]))
        self.w_h = tf.Variable(tf.random_normal([h_kernel,h_kernel,x_out,x_out]))
        self.act,self.d_act = act,d_act
        self.hidden  = tf.Variable(tf.zeros([timestamp,width_height,width_height,x_in]))
        self.hiddenA = tf.Variable(tf.zeros([timestamp,width_height,width_height,x_out]))

        self.m_x,self.h_x = tf.Variable(tf.zeros_like(self.w_x)),tf.Variable(tf.zeros_like(self.w_x))
        self.m_h,self.h_h = tf.Variable(tf.zeros_like(self.w_h)),tf.Variable(tf.zeros_like(self.w_h))
    
    def feedforward(self,input=None,timestamp=None):
        self.layer_x = tf.nn.conv2d(input,self.w_x,strides=[1,1,1,1],padding="VALID")
        self.layer_h = tf.nn.conv2d(tf.expand_dims(self.hiddenA[timestamp-1,:,:,:],axis=0),self.w_h,strides=[1,1,1,1],padding="SAME")

        hidden_assign = []
        hidden_assign.append(tf.assign(self.hidden[timestamp,:,:,:],tf.squeeze(self.layer_x+self.layer_h)))
        hidden_assign.append(tf.assign(self.hiddenA[timestamp,:,:,:],self.act(self.hidden[timestamp,:,:,:]) ))
        
        return self.hiddenA[timestamp,:,:,:],hidden_assign
        


# read the data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

train_images = np.vstack((mnist.train.images,mnist.validation.images))
train_images = np.reshape(train_images,(len(train_images),28,28,1)).astype(np.float32)
train_label  = np.vstack((mnist.train.labels,mnist.validation.labels)).astype(np.float32)

test_images = np.reshape(mnist.test.images,(len(mnist.test.images),28,28,1)).astype(np.float32)
test_label  = mnist.test.labels.astype(np.float32)

# Make class
l1 = RCNN(timestamp=5,x_in=1,x_out=3,
        x_kernel = 3,h_kernel=5,width_height=26,
        act=tf_arctan,d_act=d_tf_acrtan)


# Make Graphs
x = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

s = tf.contrib.image.rotate(x,90)

# layer1_1 = l1.feedforward(x,1)
# layer1_1 = l1.feedforward(x,1)
# layer1_1 = l1.feedforward(x,1)
# layer1_1 = l1.feedforward(x,1)




# Hyper Param
num_epoch = 1
batch_size = 1000

# Make session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        train_images,train_label = shuffle(train_images,train_label)

        for current_batch_index in range(0,len(train_images),1):
            
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_label[current_batch_index:current_batch_index+batch_size,:]

            print(current_batch.shape)
            print(current_batch_label.shape)
            
            sess_results = sess.run([s],feed_dict={x:current_batch,y:current_batch_label})
            sssss = input()

            sys.exit()
            





# -- end code --