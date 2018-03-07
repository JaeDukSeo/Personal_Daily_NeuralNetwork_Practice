import tensorflow as tf
import numpy as np,sys
from numpy import float32
import matplotlib.pyplot as plt

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return 7 

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf.log(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tansh(x): return 1.0 - tf.square(tf_tanh(x))

def gaussian_noise_layer(input_layer, std=1.0):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

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
    # return dict

def visualize_image(X,Y,names,id):
    rgb = X[:,id]
    img = rgb.reshape(3,32,32).transpose([1, 2, 0])
    plt.imshow(img)
    plt.title(names[id])
    plt.show()

# Make each class for the networks
class prepnetwork():
    
    def __init__(self):
        self.w1 = tf.Variable(tf.random_normal([3,3,3,50]))
        self.w2 = tf.Variable(tf.random_normal([3,3,50,50]))
        self.w3 = tf.Variable(tf.random_normal([3,3,50,50]))
        self.w4 = tf.Variable(tf.random_normal([3,3,50,50]))
        self.w5 = tf.Variable(tf.random_normal([3,3,50,3]))

    def getw(self): return [self.w1,self.w2,self.w3,self.w4,self.w5]

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

    def getw(self): return [self.w1,self.w2,self.w3,self.w4,self.w5]

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

    def getw(self): return [self.w1,self.w2,self.w3,self.w4,self.w5]

    def feedforward(self,input=None):
        layer1 = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        layer2 = tf.nn.conv2d(layer1,self.w2,strides=[1,1,1,1],padding='SAME')
        layer3 = tf.nn.conv2d(layer2,self.w3,strides=[1,1,1,1],padding='SAME')
        layer4 = tf.nn.conv2d(layer3,self.w4,strides=[1,1,1,1],padding='SAME')
        layer5 = tf.nn.conv2d(layer4,self.w5,strides=[1,1,1,1],padding='SAME')
        return layer5

    def backpropagation(self,gradient=None):
        return 4

# ------- Preprocess Data --------
X,Y,names = unpickle('../../z_CIFAR_data/cifar10batchespy/data_batch_1')
print(len(X))
print(len(Y))
print(len(names))

print(X.shape)
print(Y.shape)
print(names.shape)



sys.exit()

# Declare the Objects and the networks, and get the weigths for auto
prepnetwork = prepnetwork()
hididingnetwork = hididingnetwork()
revealnetwork = revealnetwork()

prepw,hidw,revw = prepnetwork.getw(),hididingnetwork.getw(),revealnetwork.getw()

# Make the Graph
s = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
c = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
beta = tf.placeholder(dtype=tf.float32)

# --- Prep part ---
layer_prep  = prepnetwork.feedforward(s)

# --- Cover part ---
cover_and_secret = tf.concat([layer_prep,c],3)
c_alpha = hididingnetwork.feedforward(cover_and_secret) 

# --- Adding Noise Part -----
c_alpha = gaussian_noise_layer(c_alpha)

# --- Reveal part ---
s_alpha = revealnetwork.feedforward(c_alpha) 

# --- cost -----
c_cost = tf.abs(c-c_alpha)
s_cost = beta * tf.abs(s-s_alpha)
total_cost = tf.reduce_sum(c_cost) + tf.reduce_sum(s_cost)

# --- Auto diff ----
auto_train = tf.train.AdamOptimizer().minimize(total_cost,var_list=prepw+hidw+revw)

# --- Back Propagation for Reveal Network ----
# grad_reveal = revealnetwork.backpropagation()


# Start the training Session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    temp = float32(np.ones((1,32,32,3)))
    sess_results = sess.run([total_cost,auto_train],feed_dict={s:temp,c:temp,beta:1.0})

    print(sess_results[0])







# -- end code --