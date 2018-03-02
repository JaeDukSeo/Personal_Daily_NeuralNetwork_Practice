import numpy as np,sys
import tensorflow as tf
from sklearn.utils import shuffle
from scipy import signal
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(678)
np.random.seed(5678)
np.set_printoptions(precision=3,suppress=True)


def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_arctan(x): return tf.atan(x)
def d_tf_arctan(x): return 1/(1+tf.square(x))

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLU(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)
    
def tf_elu(x,alpha=2): return alpha*tf.nn.elu(x)
def d_tf_leu(x,alpha=2):
    one_mask  = tf.cast(tf.greater(x, 0),dtype=tf.float32)
    zero_mask = tf_elu(tf.cast(tf.less_equal(x, 0),dtype=tf.float32) * x) + alpha
    return one_mask + zero_mask

# 0. Declare Training Data and Labels
mnnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)

# 1. Declare Class
class generator():
    
    def __init__(self):

        self.w1 = tf.Variable(tf.random_normal([7,7,1,3]))
        self.w2 = tf.Variable(tf.random_normal([5,5,3,5]))
        self.w3 = tf.Variable(tf.random_normal([3,3,5,7]))
        


class discrimator():    
    def __init__(self):
        print(4)
# 2. Make Graph

# 3. Train Session
with tf.Session() as sess: 
    sss = np.array([1,2,-9,3,-90,-342,32434,-56789765456])
    temp = tf.constant(sss,dtype=tf.float32)
    print(tf_ReLU(temp).eval())
    print(d_tf_ReLU(temp).eval())
    print(temp.eval())
    print(tf_elu(temp).eval())
    print(d_tf_leu(temp).eval())
    


# -- end code ...