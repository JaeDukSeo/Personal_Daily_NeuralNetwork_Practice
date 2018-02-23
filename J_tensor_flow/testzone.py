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
num_epoch =1201
batch_size = 1000
print_size = 100
beta1,beta2 =0.9,0.999 
adam_e = 0.00000001

# 2. Build a model with class
class simple_FCC():
    
    def __init__(self,activation=None,d_activation=None,input_dim = None,ouput_dim = None):
        
        self.w = tf.Variable(tf.random_normal([input_dim,ouput_dim],stddev=1.0))
        self.input = None
        self.output = None

        self.activation_function = activation
        self.d_activation_function = d_activation
        
        self.layer,self.layerA = None,None
        self.m, self.v = tf.Variable(tf.zeros([input_dim,ouput_dim],dtype=tf.float32,)),tf.Variable(tf.zeros([input_dim,ouput_dim],dtype=tf.float32,))

    def feed_forward(self,input=None):
        self.input = input

        self.layer =tf.matmul(input,self.w)
        self.layerA = self.output = self.activation_function(self.layer)

        return self.output

    def back_propagation(self,gradient= None,check=None):
        
        grad_part_1 = gradient
        grad_part_2 = self.d_activation_function(self.layer)
        grad_part_w = self.input

        grad_common = tf.multiply(grad_part_1,grad_part_2)
        grad_w = tf.matmul(tf.transpose(grad_part_w),grad_common)
        
        grad_pass = tf.matmul(grad_common,tf.transpose(self.w))
        update_w = [tf.assign(self.w, tf.subtract(self.w,tf.multiply(learning_rate,grad_w)) )]

        return grad_pass,update_w

    def google_brain_noise_back_prop(self,gradient= None,noise=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_activation_function(self.layer)
        grad_part_w = self.input

        grad_common = tf.multiply(grad_part_1,grad_part_2)
        grad_w = tf.matmul(tf.transpose(grad_part_w),grad_common)
        
        grad_pass = tf.matmul(grad_common,tf.transpose(self.w))
        update_w = [tf.assign(self.w, tf.subtract(self.w,tf.multiply(learning_rate,grad_w + noise)) )]

        return grad_pass,update_w

    def dilated_ADAM(self,gradient= None,check=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_activation_function(self.layer)
        grad_part_w = self.input

        grad_common = tf.multiply(grad_part_1,grad_part_2)
        grad_w = tf.matmul(tf.transpose(grad_part_w),grad_common)
        
        grad_pass = tf.matmul(grad_common,tf.transpose(self.w))

        update_w = [tf.assign(self.m,beta1*self.m  + (1.0-beta1) *  grad_w  )]
        update_w.append(tf.assign(self.v,beta2*self.v  + (1.0-beta2) * tf.square(grad_w)  ))
        
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middle = learning_rate/ (tf.sqrt(v_hat) + adam_e)

        update_w.append(tf.assign(self.w, tf.subtract(self.w,tf.multiply(adam_middle,m_hat)) ))

        return grad_pass,update_w

    def dilated_noise_ADAM(self,gradient= None,noise=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_activation_function(self.layer)
        grad_part_w = self.input

        grad_common = tf.multiply(grad_part_1,grad_part_2)
        grad_w = tf.matmul(tf.transpose(grad_part_w),grad_common)
        
        grad_pass = tf.matmul(grad_common,tf.transpose(self.w))

        update_w = [tf.assign(self.m,beta1*self.m  + (1.0-beta1) *  grad_w  )]
        update_w.append(tf.assign(self.v,beta2*self.v  + (1.0-beta2) * tf.square(grad_w)  ))
        
        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middle = learning_rate/ (tf.sqrt(v_hat) + adam_e)

        update_w.append(tf.assign(self.w, tf.subtract(self.w,tf.multiply(adam_middle,m_hat+ noise)) ))

        return grad_pass,update_w



# 81146.2851563
# proportion_rate = 800
# decay_rate = 0.6

# 91520.3007813
# proportion_rate = 800
# decay_rate = 0.2

# 81872.0473633
# proportion_rate = 800
# decay_rate = 0.8

# 91208.5732422
# proportion_rate = 600
# decay_rate = 0.4

# 81143.6395264
# proportion_rate = 900
# decay_rate = 0.7

# 81139.6497803
# proportion_rate = 999
# decay_rate = 0.777

# 81784.1019287
# proportion_rate = 999
# decay_rate = 0.0777

#  81891.7459717
# proportion_rate = 1000
# decay_rate = 0.08

# 91578.8916016
# proportion_rate = 1000
# decay_rate = 0.8888

# 81527.7155762
# proportion_rate = 900
# decay_rate = 0.07

# =============Dilated Back Propagation ======================
proportion_rate = 999
decay_rate = 0.777

# 3. Delcare the Model
layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 784, ouput_dim = 1024)
layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 1024,ouput_dim = 1024)
layer3_dil = simple_FCC(tf_tanh,d_tf_tanh,    input_dim = 1024,ouput_dim = 10)

# 4. Build a Graph
x_dil = tf.placeholder(tf.float32, shape=(None, 784))
y_dil = tf.placeholder(tf.float32, shape=(None,10))
iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

l1_dil = layer1_dil.feed_forward(x_dil)
l2_dil = layer2_dil.feed_forward(l1_dil)
l3_dil = layer3_dil.feed_forward(l2_dil)
l3Soft_dil = tf_softmax(l3_dil)

cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l3Soft_dil) + (1-y_dil) * tf.log(1-l3Soft_dil ) ) )

gradient3_dil,weightup_3_dil = layer3_dil.dilated_ADAM(l3Soft_dil - y_dil)
gradient2_dil,weightup_2_dil = layer2_dil.dilated_ADAM(gradient3_dil)
gradient1_dil,weightup_1_dil = layer1_dil.dilated_ADAM(tf.add(gradient2_dil,tf.multiply(decay_propotoin_rate,gradient3_dil)))
weight_update_dil = [weightup_3_dil,weightup_2_dil,weightup_1_dil]

total_cost = 0 
dilated_cost = []
# 4. Run the session - Manual
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch) :
        for current_image_batch in range(0,len(training_images),batch_size):
            
            current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]
            current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]

            sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })
            total_cost = total_cost +sess_result[0]
            print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0],end='\r' )

        if iter % print_size == 0 :
            print('\n--------------------')
            print("Current Iter: ", iter, " Current total cost: ",total_cost)
            current_batch = testing_images[0:0+2,:]
            current_label = testing_lables[0:0+2,:]

            sess_result = sess.run([cost_dil,l3Soft_dil],feed_dict={x_dil:current_batch,y_dil:current_label })
            print(
            'Current Predict : ',sess_result[1][0],
            'GT : ',current_label[0],'\n'
            'Current Predict : ',sess_result[1][1],
            'GT : ',current_label[1]
            )
            print('--------------------')

        dilated_cost.append(total_cost/len(training_images))
        total_cost = 0
# =============Dilated Back Propagation ======================

print('\n===== MOVING ON TO THE NEXT Goolge Brain Noise ============\n')






































# =============Dilated Back Propagation ======================
proportion_rate = 999
decay_rate = 0.777

# 3. Delcare the Model
layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 784, ouput_dim = 1024)
layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 1024,ouput_dim = 1024)
layer3_dil = simple_FCC(tf_tanh,d_tf_tanh,    input_dim = 1024,ouput_dim = 10)

# 4. Build a Graph
x_dil = tf.placeholder(tf.float32, shape=(None, 784))
y_dil = tf.placeholder(tf.float32, shape=(None,10))
iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

l1_dil = layer1_dil.feed_forward(x_dil)
l2_dil = layer2_dil.feed_forward(l1_dil)
l3_dil = layer3_dil.feed_forward(l2_dil)
l3Soft_dil = tf_softmax(l3_dil)

cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l3Soft_dil) + (1-y_dil) * tf.log(1-l3Soft_dil ) ) )

gradient3_dil,weightup_3_dil = layer3_dil.back_propagation(l3Soft_dil - y_dil)
gradient2_dil,weightup_2_dil = layer2_dil.back_propagation(gradient3_dil)
gradient1_dil,weightup_1_dil = layer1_dil.back_propagation(tf.add(gradient2_dil,tf.multiply(decay_propotoin_rate,gradient3_dil)))
weight_update_dil = [weightup_3_dil,weightup_2_dil,weightup_1_dil]

total_cost = 0 
dilated_cost2 = []
# 4. Run the session - Manual
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch) :
        for current_image_batch in range(0,len(training_images),batch_size):
            
            current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]
            current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]

            sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })
            total_cost = total_cost +sess_result[0]
            print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0],end='\r' )

        if iter % print_size == 0 :
            print('\n--------------------')
            print("Current Iter: ", iter, " Current total cost: ",total_cost)
            current_batch = testing_images[0:0+2,:]
            current_label = testing_lables[0:0+2,:]

            sess_result = sess.run([cost_dil,l3Soft_dil],feed_dict={x_dil:current_batch,y_dil:current_label })
            print(
            'Current Predict : ',sess_result[1][0],
            'GT : ',current_label[0],'\n'
            'Current Predict : ',sess_result[1][1],
            'GT : ',current_label[1]
            )
            print('--------------------')

        dilated_cost2.append(total_cost/len(training_images))
        total_cost = 0
# =============Dilated Back Propagation ======================

print('\n===== MOVING ON TO THE NEXT Goolge Brain Noise ============\n')





































# =============Dilated Back Propagation ======================
proportion_rate = 999
decay_rate = 0.777

# 3. Delcare the Model
layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 784, ouput_dim = 1024)
layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 1024,ouput_dim = 1024)
layer3_dil = simple_FCC(tf_tanh,d_tf_tanh,    input_dim = 1024,ouput_dim = 10)

# 4. Build a Graph
x_dil = tf.placeholder(tf.float32, shape=(None, 784))
y_dil = tf.placeholder(tf.float32, shape=(None,10))
iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

l1_dil = layer1_dil.feed_forward(x_dil)
l2_dil = layer2_dil.feed_forward(l1_dil)
l3_dil = layer3_dil.feed_forward(l2_dil)
l3Soft_dil = tf_softmax(l3_dil)

cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l3Soft_dil) + (1-y_dil) * tf.log(1-l3Soft_dil ) ) )

gradient3_dil,weightup_3_dil = layer3_dil.dilated_ADAM(l3Soft_dil - y_dil)
gradient2_dil,weightup_2_dil = layer2_dil.dilated_ADAM(gradient3_dil)
gradient1_dil,weightup_1_dil = layer1_dil.dilated_ADAM(tf.add(gradient2_dil,tf.multiply(decay_propotoin_rate,gradient3_dil)))
weight_update_dil = [weightup_3_dil,weightup_2_dil,weightup_1_dil]

total_cost = 0 
dilated_cost3 = []
# 4. Run the session - Manual
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch) :
        for current_image_batch in range(0,len(training_images),batch_size):
            
            current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]
            current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]

            sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })
            total_cost = total_cost +sess_result[0]
            print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0],end='\r' )

        if iter == 500: 
            proportion_rate = 1000
            decay_rate = 0.08

        if iter % print_size == 0 :
            print('\n--------------------')
            print("Current Iter: ", iter, " Current total cost: ",total_cost)
            current_batch = testing_images[0:0+2,:]
            current_label = testing_lables[0:0+2,:]

            sess_result = sess.run([cost_dil,l3Soft_dil],feed_dict={x_dil:current_batch,y_dil:current_label })
            print(
            'Current Predict : ',sess_result[1][0],
            'GT : ',current_label[0],'\n'
            'Current Predict : ',sess_result[1][1],
            'GT : ',current_label[1]
            )
            print('--------------------')

        dilated_cost3.append(total_cost/len(training_images))
        total_cost = 0
# =============Dilated Back Propagation ======================

print('\n===== MOVING ON TO THE NEXT Goolge Brain Noise ============\n')






# =============Dilated Back Propagation ======================
proportion_rate = 999
decay_rate = 0.777

# 3. Delcare the Model
layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 784, ouput_dim = 1024)
layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 1024,ouput_dim = 1024)
layer3_dil = simple_FCC(tf_tanh,d_tf_tanh,    input_dim = 1024,ouput_dim = 10)

# 4. Build a Graph
x_dil = tf.placeholder(tf.float32, shape=(None, 784))
y_dil = tf.placeholder(tf.float32, shape=(None,10))
iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

l1_dil = layer1_dil.feed_forward(x_dil)
l2_dil = layer2_dil.feed_forward(l1_dil)
l3_dil = layer3_dil.feed_forward(l2_dil)
l3Soft_dil = tf_softmax(l3_dil)

cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l3Soft_dil) + (1-y_dil) * tf.log(1-l3Soft_dil ) ) )

gradient3_dil,weightup_3_dil = layer3_dil.back_propagation(l3Soft_dil - y_dil)
gradient2_dil,weightup_2_dil = layer2_dil.back_propagation(gradient3_dil)
gradient1_dil,weightup_1_dil = layer1_dil.back_propagation(tf.add(gradient2_dil,tf.multiply(decay_propotoin_rate,gradient3_dil)))
weight_update_dil = [weightup_3_dil,weightup_2_dil,weightup_1_dil]

total_cost = 0 
dilated_cost4 = []
# 4. Run the session - Manual
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch) :
        for current_image_batch in range(0,len(training_images),batch_size):
            
            current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]
            current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]

            sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })
            total_cost = total_cost +sess_result[0]
            print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0],end='\r' )

        if iter == 500: 
            proportion_rate = 1000
            decay_rate = 0.08

        if iter % print_size == 0 :
            print('\n--------------------')
            print("Current Iter: ", iter, " Current total cost: ",total_cost)
            current_batch = testing_images[0:0+2,:]
            current_label = testing_lables[0:0+2,:]

            sess_result = sess.run([cost_dil,l3Soft_dil],feed_dict={x_dil:current_batch,y_dil:current_label })
            print(
            'Current Predict : ',sess_result[1][0],
            'GT : ',current_label[0],'\n'
            'Current Predict : ',sess_result[1][1],
            'GT : ',current_label[1]
            )
            print('--------------------')

        dilated_cost4.append(total_cost/len(training_images))
        total_cost = 0
# =============Dilated Back Propagation ======================

print('\n===== MOVING ON TO THE NEXT Goolge Brain Noise ============\n')










# =============Dilated Back Propagation ======================
proportion_rate = 999
decay_rate = 0.777

# 3. Delcare the Model
layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 784, ouput_dim = 1024)
layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 1024,ouput_dim = 1024)
layer3_dil = simple_FCC(tf_tanh,d_tf_tanh,    input_dim = 1024,ouput_dim = 10)

# 4. Build a Graph
x_dil = tf.placeholder(tf.float32, shape=(None, 784))
y_dil = tf.placeholder(tf.float32, shape=(None,10))
iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

l1_dil = layer1_dil.feed_forward(x_dil)
l2_dil = layer2_dil.feed_forward(l1_dil)
l3_dil = layer3_dil.feed_forward(l2_dil)
l3Soft_dil = tf_softmax(l3_dil)

cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l3Soft_dil) + (1-y_dil) * tf.log(1-l3Soft_dil ) ) )

gradient3_dil,weightup_3_dil = layer3_dil.back_propagation(l3Soft_dil - y_dil)
gradient2_dil,weightup_2_dil = layer2_dil.back_propagation(gradient3_dil)
gradient1_dil,weightup_1_dil = layer1_dil.back_propagation(tf.add(gradient2_dil,tf.multiply(decay_propotoin_rate,gradient3_dil)))
weight_update_dil = [weightup_3_dil,weightup_2_dil,weightup_1_dil]

total_cost = 0 
dilated_cost5 = []
# 4. Run the session - Manual
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch) :
        for current_image_batch in range(0,len(training_images),batch_size):
            
            current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]
            current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]

            sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })
            total_cost = total_cost +sess_result[0]
            print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0],end='\r' )

        if iter == 500: 
            proportion_rate = 800
            decay_rate = 0.8

        if iter % print_size == 0 :
            print('\n--------------------')
            print("Current Iter: ", iter, " Current total cost: ",total_cost)
            current_batch = testing_images[0:0+2,:]
            current_label = testing_lables[0:0+2,:]

            sess_result = sess.run([cost_dil,l3Soft_dil],feed_dict={x_dil:current_batch,y_dil:current_label })
            print(
            'Current Predict : ',sess_result[1][0],
            'GT : ',current_label[0],'\n'
            'Current Predict : ',sess_result[1][1],
            'GT : ',current_label[1]
            )
            print('--------------------')

        dilated_cost5.append(total_cost/len(training_images))
        total_cost = 0
# =============Dilated Back Propagation ======================

print('\n===== MOVING ON TO THE NEXT Goolge Brain Noise ============\n')





















# =============Auto Back Propagation GradientDescentOptimizer ======================
# 3. Delcare the Model
layer1_auto = simple_FCC(tf_log,d_tf_log,input_dim = 784,ouput_dim = 1024)
layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,input_dim = 1024,ouput_dim = 1024)
layer3_auto = simple_FCC(tf_tanh,d_tf_tanh,input_dim = 1024,ouput_dim = 10)

# 4. Build a Graph
x_auto = tf.placeholder(tf.float32, shape=(None, 784))
y_auto = tf.placeholder(tf.float32, shape=(None,10))

l1_auto = layer1_auto.feed_forward(x_auto)
l2_auto = layer2_auto.feed_forward(l1_auto)
l3_auto = layer3_auto.feed_forward(l2_auto)
l3Soft_auto = tf_softmax(l3_auto)

cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l3Soft_auto) + (1-y_auto) * tf.log(1-l3Soft_auto ) ) )
auto_dif = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost_auto)

total_cost = 0 
auto_cost = []
# 4. Run the session - Manual
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch) :
        for current_image_batch in range(0,len(training_images),batch_size):
            
            current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]
            current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]

            sess_result = sess.run([cost_auto,auto_dif],feed_dict={x_auto:current_batch,y_auto:current_label })
            total_cost = total_cost +sess_result[0]
            print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0],end='\r' )

        if iter % print_size == 0 :
            print('\n--------------------')
            print("Current Iter: ", iter, " Current total cost: ",total_cost)
            current_batch = testing_images[0:0+2,:]
            current_label = testing_lables[0:0+2,:]

            sess_result = sess.run([cost_auto,l3Soft_auto],feed_dict={x_auto:current_batch,y_auto:current_label })
            print(
            'Current Predict : ',sess_result[1][0],
            'GT : ',current_label[0],'\n'
            'Current Predict : ',sess_result[1][1],
            'GT : ',current_label[1]
            )
            print('--------------------')

        auto_cost.append(total_cost/len(training_images))
        total_cost = 0
# =============Auto Back Propagation GradientDescentOptimizer ======================

print('\n===== MOVING ON TO THE NEXT============\n')































plt.plot(range(len(dilated_cost)),dilated_cost,color='r',label='Dilated Noise')
plt.plot(range(len(dilated_cost)),dilated_cost2,color='g',label='Dilated Noise 2')
plt.plot(range(len(dilated_cost)),dilated_cost3,color='b',label='Dilated Noise 3')
plt.plot(range(len(dilated_cost)),dilated_cost4,color='m',label='Dilated Noise 4')
plt.plot(range(len(dilated_cost)),dilated_cost5,color='c',label='Dilated Noise 5')


plt.plot(range(len(dilated_cost)),auto_cost,color='y',label='auto adam')
plt.legend()
plt.show()
