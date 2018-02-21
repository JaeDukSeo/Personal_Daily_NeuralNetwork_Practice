import numpy as np,sys
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(678)
np.set_printoptions(2)
tf.set_random_seed(678)

def tf_log(x=None):
    return tf.sigmoid(x)
def d_tf_log(x=None):
    return tf_log(x) * (1.0 - tf_log(x))

def tf_tanh(x=None):
    return tf.tanh(x)
def d_tf_tanh(x=None):
    return 1.0 - tf.square(tf.tanh(x))

def tf_arctan(x =None):
    return tf.atan(x)
def d_tf_arctan(x=None):
    return 1.0/(1.0+tf.square(x))

def tf_softmax(x=None):
    return tf.nn.softmax(x)

# 1. Preprocess the data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
testing_images, testing_lables =mnist.test.images,mnist.test.labels
training_images,training_lables =mnist.train.images,mnist.train.labels

# 2. Global 
learning_rate= 0.001
num_epoch =501
batch_size = 1000
print_size = 100
