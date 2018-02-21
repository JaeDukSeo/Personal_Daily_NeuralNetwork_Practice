import numpy as np,sys
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(678)
tf.set_random_seed(678)

def tf_log(x):
    return tf.sigmoid(x)
def d_tf_log(x):
    return tf_log(x) * (1.0 - tf_log(x))

def tf_tanh(x):
    return tf.tanh(x)
def d_tf_tanh(x):
    return 1.0 - tf.square(tf.tanh(x))


# 1. Preprocess the data


# 2. Build a model with class
class simple_FCC():
    
    def __init__(self):
        
        w = tf.Variable(tf.random_normal([784,784],stddev=0.45))

    def feed_forward(self):
        print(8)

# 3. Delcare the Model
layer1 = simple_FCC()

# 4. Run the session
with tf.Session() as sess:

    value = sess.run(d_tf_tanh(7.9))
    print(value)

    print("start over")














# -- end code ---