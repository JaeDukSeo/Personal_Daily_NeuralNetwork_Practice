import tensorflow as tf
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle

np.random.seed(678)
tf.set_random_seed(6786)

def tf_Relu(x): return tf.nn.relu(x)
def d_tf_ReLu(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_arctan(x):return tf.atan(x)
def d_tf_arctan(x):return 1.0/(1+tf.square(x))

def tf_softmax(x): return tf.nn.softmax(x)
# =============== Create Class ===========
class contextLayer():
    
    def __init__(self,kernelsize,inchannel,outchannel):
        self.w = tf.Variable(tf.random_normal([kernelsize,kernelsize,inchannel,outchannel]))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
    def getw(self): return self.w

    def feedforward(self,input=None,dilationfactor=None,Same=False):
        self.input = input
        self.layer  = tf.nn.atrous_conv2d(input,self.w, rate=dilationfactor,padding="SAME")
        self.layerA = tf_Relu(self.layer)
        return self.layerA

    def backprop(self,gradient=None,dilation_factor = None):
        grad_part_1 = gradient
        grad_part_2 = d_tf_ReLu(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.transpose(tf.multiply(grad_part_1,grad_part_2),[0,2,1,3])

        grad = tf.nn.conv2d_backprop_filter(input=grad_part_3,filter_sizes=self.w.shape,
        out_backprop=grad_middle,strides=[1,1,1,1],padding='SAME',
        dilations = [1,dilation_factor,dilation_factor,1]
        )
        
        pass_size = list(self.input.shape[1:])
        pass_on_grad = tf.nn.conv2d_backprop_input(input_sizes=[batch_size]+pass_size,filter=self.w,
        out_backprop=grad_middle,strides=[1,1,1,1],padding='SAME',
        dilations = [1,dilation_factor,dilation_factor,1]
        )

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return pass_on_grad,grad_update

class FCNN():
    
    def __init__(self,input,output):
        self.w = tf.Variable(tf.random_normal([input,output]))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
    def getw(self): return self.w
    def feedforward(self,input):
        self.input = input
        self.layer  = tf.matmul(input,self.w)
        self.layerA =  tf_arctan(self.layer)
        return self.layerA
    def backprop(self,gradient=None):
        grad_part_1 = gradient
        grad_part_2 = d_tf_arctan(self.layer)
        grad_part_3 = self.input

        grad = tf.matmul(tf.transpose(grad_part_3),tf.multiply(grad_part_1,grad_part_2))
        pass_on_grad = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))
        
        return pass_on_grad,grad_update
        
# =============== Create Class =========== 

# Process Data
mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
testing_images, testing_lables =mnist.test.images,mnist.test.labels
training_images,training_lables =mnist.train.images,mnist.train.labels

testing_images = np.reshape(testing_images,  (10000,28,28,1))
training_images = np.reshape(training_images,(55000,28,28,1))

# Hyper Parameters
num_epoch = 100
batch_size = 100
learning_rate = 0.0025
print_size = 5

beta1,beta2 = 0.9,0.999
adam_e = 0.00000001

proportion_rate = 1500
decay_rate = 0.08

# Declare Models 
layer1 = contextLayer(3,1,1)
layer2 = contextLayer(3,1,1)
layer3 = contextLayer(3,1,1)
layer4 = contextLayer(3,1,1)

layer5 = contextLayer(3,1,1)
layer6 = contextLayer(3,1,1)
layer7 = contextLayer(3,1,1)
layer8 = contextLayer(3,1,1)

layer9  = FCNN(28*28,512)
layer10  =FCNN(512,128)
layer11 = FCNN(128,10)

l1w,l2w,l3w,l4w = layer1.getw(),layer2.getw(),layer3.getw(),layer4.getw()
l5w,l6w,l7w,l8w = layer5.getw(),layer6.getw(),layer7.getw(),layer8.getw()
l9w,l10w,l11w = layer9.getw(),layer10.getw(),layer11.getw()

# Create Graph
x = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

iter_variable_dil = tf.placeholder(tf.float32, shape=())
decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

l1 = layer1.feedforward(x,1)
l2 = layer2.feedforward(l1,1)
l3 = layer3.feedforward(l2,2)
l4 = layer4.feedforward(l3,2)

l5 = layer5.feedforward(l4,4)
l6 = layer6.feedforward(l5,8)
l7 = layer7.feedforward(l6,16)
l8 = layer8.feedforward(l7,1)

l9Input = tf.reshape(l8,(batch_size,-1))
l9  = layer9.feedforward(l9Input)
l10 = layer10.feedforward(l9)
l11 = layer11.feedforward(l10)

final_soft = tf_softmax(l11)

cost = -1.0 * ( y*tf.log(final_soft) + (1-y) * tf.log(1-final_soft))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Auto Train
# auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=[l1w,l2w,l3w,l4w,l5w,l6w,l7w,l8w,l9w,l10w,l11w])

# manual back prop
grad_11,grad_11_u = layer11.backprop(final_soft-y)
grad_10,grad_10_u = layer10.backprop(grad_11)
grad_9,grad_9_u =   layer9.backprop(grad_10)

grad_8_Input = tf.reshape(grad_9,(batch_size,28,28,1))
grad_8,grad_8_u =   layer8.backprop(grad_8_Input,1)
grad_7,grad_7_u =   layer7.backprop(grad_8,1)
grad_6,grad_6_u =   layer6.backprop(grad_7,3)
grad_5,grad_5_u =   layer5.backprop(grad_6+decay_propotoin_rate*(grad_8+grad_7),3)

grad_4,grad_4_u =   layer4.backprop(grad_5+decay_propotoin_rate*(grad_7+grad_6),2)
grad_3,grad_3_u =   layer3.backprop(grad_4+decay_propotoin_rate*(grad_6+grad_5),2)
grad_2,grad_2_u =   layer2.backprop(grad_3+decay_propotoin_rate*(grad_5+grad_4),1)
grad_1,grad_1_u =   layer1.backprop(grad_2+decay_propotoin_rate*(grad_4+grad_3),1)

grad_update_list = [grad_11_u+grad_10_u+grad_10_u+\
                    grad_8_u+grad_7_u+grad_6_u+grad_5_u+\
                    grad_4_u+grad_3_u+grad_2_u+grad_1_u]

# Create Session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    avg_acc, avg_cost = 0,0
    avg_acc_overtime,avg_cost_overtime  = [],[]

    avg_acc_text, avg_cost_text = 0,0
    avg_acc_overtime_text,avg_cost_overtime_text  = [],[]

    for iter in range(num_epoch):
        
        # every iter shuffle the train set
        training_images,training_lables = shuffle(training_images,training_lables)

        # Run the Train Images
        for current_batch_index in range(0,len(training_images),batch_size):
            current_batch = training_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = training_lables[current_batch_index:current_batch_index+batch_size,:]

            # Commented out the auto train methods
            # sess_results = sess.run([cost,accuracy,auto_train],feed_dict={x:current_batch,y:current_batch_label})
            sess_results = sess.run([cost,accuracy,correct_prediction,grad_update_list],feed_dict={x:current_batch,y:current_batch_label,iter_variable_dil:iter })
            
            print("Current Iter: ",iter, " Current Batch Index : ", current_batch_index, " Current cost: ", np.sum(sess_results[0])," Current Acc: ",sess_results[1],  end='\r')
            avg_acc = avg_acc + np.sum(sess_results[0])/batch_size
            avg_cost = avg_cost + sess_results[1]
            
        # Run the Test Images
        for current_batch_index in range(0,len(testing_images),batch_size):
            
            current_batch = testing_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = testing_lables[current_batch_index:current_batch_index+batch_size,:]
            sess_results = sess.run([cost,accuracy,correct_prediction],feed_dict={x:current_batch,y:current_batch_label })
            
            avg_acc_text = avg_acc_text + np.sum(sess_results[0])/batch_size
            avg_cost_text = avg_cost_text + sess_results[1]

        # if Print size 
        if iter%print_size==0:
            print('\n===========================')
            print("Current Avg Accuracy for Train Images: ",avg_cost/(len(training_images)/batch_size) )
            print("Current Avg Cost for Train Images: ",avg_acc/(len(training_images)/batch_size))
            print("Current Avg Accuracy for Test Images: ",avg_cost_text/(len(testing_images)/batch_size))
            print("Current Avg Cost for Test Images: ",avg_acc_text/(len(testing_images)/batch_size))
            print("========================\n")            

        avg_acc_overtime.append(avg_acc/(len(training_images)/batch_size))
        avg_cost_overtime.append(avg_cost/(len(training_images)/batch_size))
        avg_acc, avg_cost = 0,0
        
        avg_acc_overtime_text.append(avg_acc_text/(len(testing_images)/batch_size))
        avg_cost_overtime_text.append(avg_cost_text/(len(testing_images)/batch_size))
        avg_acc_text, avg_cost_text = 0,0



# -- end code --