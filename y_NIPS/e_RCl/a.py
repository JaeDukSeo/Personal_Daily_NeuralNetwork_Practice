import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(678)
tf.set_random_seed(5678)

# activation
def tf_log(x): return tf.Sigmoid(x)
def d_tf_log(x): return tf_log(x) * ( 1.0 - tf_log(x))

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLU(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_arctan(x): return tf.nn.atan(x)
def d_tf_acrtan(x): return 1/(1 + tf.square(x))

def tf_softmax(x): return tf.nn.softmax(x)

# Make Class
class RCNN():
    
    def __init__(self,timestamp,x_in,x_out,x_kernel,h_kernel,width_height):
        
        self.w_x = tf.Variable(tf.random_normal([x_kernel,x_kernel,x_in,x_out]))
        self.w_h = tf.Variable(tf.random_normal([h_kernel,h_kernel,x_out,x_out]))

        self.hidden  = tf.Variable(tf.zeros([timestamp,width_height,width_height,1]))
        self.hiddenA = tf.Variable(tf.zeros([timestamp,width_height,width_height,1]))



# read the data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

train_images = np.vstack((mnist.train.images,mnist.validation.images))
train_images = np.reshape(train_images,(len(train_images),28,28,1))
train_label  = np.vstack((mnist.train.labels,mnist.validation.labels))

test_images = np.reshape(mnist.test.images,(len(mnist.test.images),28,28,1))
test_label  = mnist.test.labels





print(train_images.shape)
print(train_label.shape)

print(test_images.shape)
print(test_label.shape)




# -- end code --