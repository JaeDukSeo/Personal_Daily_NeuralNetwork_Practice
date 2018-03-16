import tensorflow as tf
import numpy as np,sys
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

np.random.seed(678)
tf.set_random_seed(5678)

# activation
def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * ( 1.0 - tf_log(x))

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLU(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_arctan(x): return tf.atan(x)
def d_tf_acrtan(x): return 1/(1 + tf.square(x))

def tf_softmax(x): return tf.nn.softmax(x)

# Different noises
def gaussian_noise_layer(input_layer):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=1.0, dtype=tf.float32) 
    return input_layer + noise

def possin_layer(layer):
    noise = tf.random_poisson(lam=0.0,shape=tf.shape(layer),dtype=tf.float32)
    return noise + layer

def uniform_layer(input_layer):
    noise = tf.random_uniform(shape=tf.shape(input_layer),dtype=tf.float32)
    return noise + input_layer

def gamma_layer(input_layer):
    noise = tf.random_gamma(shape=tf.shape(input_layer),alpha=0.0,dtype=tf.float32)
    return noise + input_layer

# Make Class
class FNN():
    
    def __init__(self,input,output,act,d_act):
        self.w = tf.Variable(tf.random_normal([input,output]))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input):
        self.input = input
        self.layer  = tf.matmul(input,self.w)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient):
        grad_part_1 = gradient 
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad = tf.matmul(tf.transpose(grad_part_3),tf.multiply(grad_part_1,grad_part_2))
        grad_pass = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        update_w = []
        update_w.append( tf.assign(self.m,beta_1*self.m + (1-beta_1) * grad)  )
        update_w.append( tf.assign(self.v,beta_2*self.v + (1-beta_2) * grad ** 2) )

        m_hat = self.m/(1-beta_1)
        v_hat = self.v/(1-beta_2)

        adam_middle = learning_rate/(tf.sqrt(v_hat) + adam_e)
        update_w.append( tf.assign(self.w, tf.subtract(self.w,adam_middle*m_hat))  )

        return grad_pass,update_w
        

class RCNN():
    
    def __init__(self,timestamp,
                x_in,x_out,
                x_kernel,h_kernel,width_height,
                act,d_act,batch_size):
        
        self.w_x = tf.Variable(tf.random_normal([x_kernel,x_kernel,x_in,x_out]))
        self.w_h = tf.Variable(tf.random_normal([h_kernel,h_kernel,x_out,x_out]))
        self.act,self.d_act = act,d_act

        self.hidden  = tf.Variable(tf.zeros([timestamp,batch_size,width_height,width_height,x_out]))
        self.hiddenA = tf.Variable(tf.zeros([timestamp,batch_size,width_height,width_height,x_out]))

        self.m_x,self.v_x = tf.Variable(tf.zeros_like(self.w_x)),tf.Variable(tf.zeros_like(self.w_x))
        self.m_h,self.v_h = tf.Variable(tf.zeros_like(self.w_h)),tf.Variable(tf.zeros_like(self.w_h))
    
    def feedforward(self,input=None,timestamp=None):
        self.layer_x = tf.nn.conv2d(input,self.w_x,strides=[1,1,1,1],padding="VALID")
        self.layer_h = tf.nn.conv2d(self.hiddenA[timestamp-1,:,:,:,:],self.w_h,strides=[1,1,1,1],padding="SAME")

        hidden_assign = []
        hidden_assign.append( tf.assign( self.hidden[timestamp,:,:,:,:],  self.layer_x+self.layer_h))
        hidden_assign.append( tf.assign(self.hiddenA[timestamp,:,:,:,:], self.act(self.hidden[timestamp,:,:,:,:]) ))
        
        return self.hiddenA[timestamp,:,:,:,:],hidden_assign
        


# read the data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

train_images = np.vstack((mnist.train.images,mnist.validation.images))
train_images = np.reshape(train_images,(len(train_images),28,28,1)).astype(np.float32)
train_label  = np.vstack((mnist.train.labels,mnist.validation.labels)).astype(np.float32)

test_images = np.reshape(mnist.test.images,(len(mnist.test.images),28,28,1)).astype(np.float32)
test_label  = mnist.test.labels.astype(np.float32)

# Hyper Param
num_epoch = 1
batch_size = 1000
learning_rate = 0.0001

beta_1,beta_2 = 0.9,0.999
adam_e = 0.00000001

# Make class
l1 = RCNN(timestamp=5,x_in=1,x_out=3,
        x_kernel = 5,h_kernel=3,width_height=24,
        act=tf_ReLU,d_act=d_tf_ReLU,batch_size=batch_size)

l2 = RCNN(timestamp=5,x_in=3,x_out=1,
        x_kernel=5,h_kernel=3,width_height=20,
        act=tf_ReLU,d_act=d_tf_ReLU,batch_size=batch_size)
    
l3 = FNN(400,1024,tf_arctan,d_tf_acrtan)
l4 = FNN(1024,2048,tf_arctan,d_tf_acrtan)
l5 = FNN(2048,10,tf_log,d_tf_log)

# Make Graphs
x = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

x1 = gaussian_noise_layer(x)
x2 = possin_layer(x)
x3 = uniform_layer(x)
x4 = gamma_layer(x)

layer_assign,backprop_assign = [],[]

layer1_1,l1_1a = l1.feedforward(x1,1)
layer1_2,l1_2a = l1.feedforward(x2,2)
layer1_3,l1_3a = l1.feedforward(x3,3)
layer1_4,l1_4a = l1.feedforward(x4,4)
layer_assign.append(l1_1a+l1_2a+l1_3a+l1_4a)

layer2_1,l2_1a = l2.feedforward(layer1_1,1)
layer2_2,l2_2a = l2.feedforward(layer1_2,2)
layer2_3,l2_3a = l2.feedforward(layer1_3,3)
layer2_4,l2_4a = l2.feedforward(layer1_4,4)
layer_assign.append(l2_1a+l2_2a+l2_3a+l2_4a)

layer3_Input = tf.reshape(layer2_4,[batch_size,-1])
layer3 = l3.feedforward(layer3_Input)
layer4 = l4.feedforward(layer3)
layer5 = l5.feedforward(layer4)

final_layer = tf_softmax(layer5)

cost = -1.0 * (y*tf.log(final_layer) + (1-y) * tf.log(1-y))
correct_prediction = tf.equal(tf.argmax(final_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# auto train 
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

grad_5,grad5w = l5.backprop(final_layer-y)
grad_4,grad4w = l4.backprop(grad_5)
grad_3,grad3w = l3.backprop(grad_4)

backprop_assign.append(grad5w+grad4w+grad3w)

# Make session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        train_images,train_label = shuffle(train_images,train_label)

        for current_batch_index in range(0,len(train_images),1):
            
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_label[current_batch_index:current_batch_index+batch_size,:]

            sess_results = sess.run([grad_5,layer_assign,backprop_assign],feed_dict={x:current_batch,y:current_batch_label})

            print(sess_results[0].shape)
            print(sess_results[1].shape)
            

            sys.exit()
            





# -- end code --