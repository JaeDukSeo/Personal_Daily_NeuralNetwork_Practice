import tensorflow as tf
import numpy as np,sys
from numpy import float32
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(678)
tf.set_random_seed(678)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return 7 

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf.log(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

def tf_softmax(x): return tf.nn.softmax(x)

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


class RCNN():
    
    def __init__(self,time_seq=None,Height=None,width=None,
                input_dim=None, hidden_dim =None,batch_size=None,act=None,d_act=None):
        
        self.w_x = tf.Variable(tf.random_normal([3,3,input_dim,hidden_dim]))
        self.w_h = tf.Variable(tf.random_normal([3,3,hidden_dim,hidden_dim]))

        self.hidden  = tf.Variable(tf.zeros([time_seq,batch_size,Height,width,hidden_dim]))
        self.hiddenA = tf.Variable(tf.zeros([time_seq,batch_size,Height,width,hidden_dim]))

        self.act,self.d_act = act,d_act

    def getw(self): return [self.w_x,self.w_h]

    def feedforward(self,input=None,timestamp=None):
        
        self.layer  = tf.nn.conv2d(input,self.w_x,strides=[1,1,1,1],padding='SAME')  + \
                      tf.nn.conv2d(self.hiddenA[timestamp,:,:,:,:], self.w_h,strides=[1,1,1,1],padding='SAME') 
        self.layerA = self.act(self.layer)

        assign_hidden = []
        assign_hidden.append(tf.assign(self.hidden[timestamp+1,:,:,:,:],self.layer))
        assign_hidden.append(tf.assign(self.hiddenA[timestamp+1,:,:,:,:],self.layerA))

        return self.layerA,assign_hidden

class FNN():
    
    def __init__(self,input_shape=None,output_shape=None):
        self.w = tf.Variable(tf.random_normal(shape=[input_shape,output_shape]))
    
    def getw(self): return [self.w]

    def feedforward(self,input=None):
        self.layer  = tf.matmul(input,self.w)
        self.layerA = tf_log(self.layer)
        return self.layerA

# ------- Preprocess Data --------
mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
testing_images, testing_lables =mnist.test.images,mnist.test.labels
training_images,training_lables =mnist.train.images,mnist.train.labels
testing_images = np.reshape(testing_images,(10000,28,28,1))
training_images = np.reshape(training_images,(55000,28,28,1))

# --- Hyper Parameter ----
batch_size = 50
num_epoch = 1000
print_size = 10

# ---- Make Objects ----
l1_RCNN = RCNN(4,28,28,1,3,batch_size,tf_tanh,d_tf_tanh)
l2_RCNN = RCNN(4,28,28,3,5,batch_size,tf_Relu,d_tf_Relu)
l3_RCNN = RCNN(4,28,28,5,1,batch_size,tf_log,d_tf_log)

l1w,l2w,l3w = l1_RCNN.getw(),l2_RCNN.getw(),l3_RCNN.getw()

# ---- Make Graph -----
x = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32)
time_stamp = tf.constant(0)
layer_uodate = []

l1_1,l1_1w = l1_RCNN.feedforward(x,time_stamp+0)
l1_2,l1_2w = l1_RCNN.feedforward(x,time_stamp+1)
l1_3,l1_3w = l1_RCNN.feedforward(x,time_stamp+2)
layer_uodate.append(l1_1w+l1_2w+l1_3w)

l2_1,l2_1w = l2_RCNN.feedforward(l1_1,time_stamp+0)
l2_2,l2_2w = l2_RCNN.feedforward(l1_2,time_stamp+1)
l2_3,l2_3w = l2_RCNN.feedforward(l1_3,time_stamp+2)
layer_uodate.append(l2_1w+l2_2w+l2_3w)

l3_1,l3_1w = l3_RCNN.feedforward(l2_1,time_stamp+0)
l3_2,l3_2w = l3_RCNN.feedforward(l2_2,time_stamp+1)
l3_3,l3_3w = l3_RCNN.feedforward(l2_3,time_stamp+2)
layer_uodate.append(l3_1w+l3_2w+l3_3w)

cost = tf.reduce_sum(tf.square(l3_3-y))

# --- Auto Train ----
auto_train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost,var_list=l1w+l2w+l3w)

# --- train sesion ----
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    total_cost_track = 0
    cost_over_time = []
    avg_accuracy = 0
    avg_cost = 0

    for iter in range(num_epoch):
        
        for batch_size_index in range(0,len(training_images),batch_size):
            current_data = training_images[batch_size_index:batch_size_index+batch_size,:,:,:]
            current_data_noise =  current_data + 0.1 * current_data.max() *np.random.randn(current_data.shape[0],current_data.shape[1],current_data.shape[2],current_data.shape[3])
            sess_result = sess.run([cost,layer_uodate,auto_train],feed_dict={x:current_data,y:current_data_noise})
            print("Current Iter : ",iter, " current batch: ",batch_size_index,  ' Current cost: ', sess_result[0],end='\r')
            total_cost_track = total_cost_track + sess_result[0]

        if iter % print_size ==0:
            current_data = testing_images[0:0+batch_size,:,:,:]
            current_data_noise =  current_data + 0.1 * current_data.max() *np.random.randn(current_data.shape[0],current_data.shape[1],current_data.shape[2],current_data.shape[3])
            sess_result = sess.run([l3_3,layer_uodate],feed_dict={x:current_data,y:current_data_noise})
            
            plt.imshow(current_data[3,:,:,0],cmap='gray')
            plt.savefig(str(iter)+'1_.png')

            plt.imshow(current_data_noise[3,:,:,0],cmap='gray')
            plt.savefig(str(iter)+'2_.png')

            plt.imshow(sess_result[0][3,:,:,0],cmap='gray')
            plt.savefig(str(iter)+'3_.png')

            print('\n------------\n')



        cost_over_time.append(total_cost_track)
        total_cost_track = 0