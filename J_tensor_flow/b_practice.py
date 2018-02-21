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
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True).test
images,label = shuffle(mnist.images,mnist.labels)
test_image_num,training_image_num = 50,1000
num_epoch = 1
cost_array = []
batch_size = 3
total_cost = 0 
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]

learning_rate= 0.001
num_epoch =300
batch_size = 100
# 2. Build a model with class
class simple_FCC():
    
    def __init__(self,activation=None,d_activation=None,input_dim = None,ouput_dim = None):
        
        self.w = tf.Variable(tf.random_normal([input_dim,ouput_dim],stddev=1.0))
        # self.b = tf.Variable(tf.random_normal([1,1],stddev=0.4))
        self.input = None
        self.output = None

        self.activation_function = activation
        self.d_activation_function = d_activation
        
        self.layer,self.layerA = None,None

    def feed_forward(self,input=None):
        self.input = input

        self.layer =tf.matmul(input,self.w)
        self.layerA = self.output = self.activation_function(self.layer)

        return self.output

    def back_propagation(self,gradient= None,check=None):
        
        grad_part_1 = gradient
        grad_part_2 = self.d_activation_function(self.layer)
        grad_part_w = self.input
        # grad_part_b = tf.ones([1,self.w.shape[1]])

        grad_common = tf.multiply(grad_part_1,grad_part_2)
        grad_w = tf.matmul(tf.transpose(grad_part_w),grad_common)
        # grad_b = tf.matmul(grad_part_b,tf.transpose(grad_common))
        
        grad_pass = tf.matmul(grad_common,tf.transpose(self.w))
        update_w = [tf.assign(self.w, tf.subtract(self.w,tf.multiply(learning_rate,grad_w)) )]
        # tf.assign(self.b,tf.subtract(self.b,tf.multiply(learning_rate,grad_b)) )]

        return grad_pass,update_w

# 3. Delcare the Model
layer1 = simple_FCC(tf_log,d_tf_log,input_dim = 784,ouput_dim = 825)
layer2 = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 825,ouput_dim = 1024)
layer3 = simple_FCC(tf_tanh,d_tf_tanh,input_dim = 1024,ouput_dim = 10)

temp = np.ones((1,784))
temp2 = np.ones((1,10))

# 4. Build a Graph
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None,10))

l1 = layer1.feed_forward(x)
l2 = layer2.feed_forward(l1)
l3 = layer3.feed_forward(l2)
l3Soft = tf_softmax(l3)

cost = tf.reduce_sum( -1 * ( y*tf.log(l3Soft) + (1-y) * tf.log(1-l3Soft ) ) )
auto_dif = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

gradient3,weightup_3 = layer3.back_propagation(l3Soft - y,3)
gradient2,weightup_2 = layer2.back_propagation(gradient3,2)
gradient1,weightup_1 = layer1.back_propagation(gradient2,1)
weight_update = [weightup_3,weightup_2,weightup_1]

total_cost = 0 

# 4. Run the session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch) :
        for current_image_batch in range(0,len(training_images),batch_size):
            
            current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]
            current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]

            # sess_result = sess.run([cost,gradient1,weight_update],feed_dict={x:current_batch,y:current_label })
            sess_result = sess.run([cost,auto_dif],feed_dict={x:current_batch,y:current_label })
            total_cost = total_cost +sess_result[0]
            print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0],end='\r' )
            

        if iter % 10 == 0 :
            print('\n--------------------')
            print("Current Iter: ", iter, " Current total cost: ",total_cost)
            current_batch = testing_images[0:0+2,:]
            current_label = testing_lables[0:0+2,:]

            sess_result = sess.run([cost,l3Soft],feed_dict={x:current_batch,y:current_label })
            print(
            'Current Predict : ',sess_result[1][0],
            'GT : ',current_label[0],'\n'
            'Current Predict : ',sess_result[1][1],
            'GT : ',current_label[1]
            )
            print('--------------------')

        total_cost = 0













# -- end code ---