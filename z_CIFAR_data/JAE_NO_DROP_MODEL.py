import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from six.moves import cPickle as pickle
from read_10_data import get_data

np.random.seed(789)
tf.set_random_seed(789)

def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf_elu(x) + 1.0 

def tf_softmax(x): return tf.nn.softmax(x)

#Accuracy measurment for the results
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# get data
NUM_CLASSES = 10
train_images, train_labels, test_images,test_labels = get_data()

# =========== Layer Class ===========
class CNNLayer():
      
  def __init__(self,kernel,in_c,out_c):
    self.w = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c],stddev=0.05,mean=0.0))
    self.m = tf.Variable(tf.zeros_like(self.w)) 

  def feedforward(self,input):
    self.input = input
    self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME')
    self.layerA = tf_elu(self.layer)
    return self.layerA

  def feedforward_avg(self,input):
    self.input = input
    self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME')
    self.layerA = tf_elu(self.layer)
    self.layerMean = tf.nn.avg_pool(self.layerA, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return self.layerMean

  def backprop(self,gradient):
    grad_part_1 = gradient
    grad_part_2 = d_tf_elu(self.layer)
    grad_part_3 = self.input

    grad_middle = tf.multiply(grad_part_1,grad_part_2)

    grad = tf.nn.conv2d_backprop_filter(
      input = grad_part_3,
      filter_sizes = self.w.shape,
      out_backprop = grad_middle,
      strides = [1,1,1,1],padding='SAME'
    )

    pass_size = list(self.input.shape[1:])
    grad_pass = tf.nn.conv2d_backprop_input(
      input_sizes =[batch_size]+pass_size,
      filter = self.w,
      out_backprop = grad_middle,
      strides = [1,1,1,1],padding='SAME'
    )

    update_w = []
    tf.assign(self.m, 0.7*self.m + learning_rate*grad)    
    tf.assign(self.w,self.w - self.m)
    return grad_pass,update_w

  def backprop_avg(self,gradient):
    grad_part_1 = tf.tile(gradient, [1,2,2,1])
    grad_part_2 = d_tf_elu(self.layer)
    grad_part_3 = self.input

    grad_middle = tf.multiply(grad_part_1,grad_part_2)

    grad = tf.nn.conv2d_backprop_filter(
      input = grad_part_3,
      filter_sizes = self.w.shape,
      out_backprop = grad_middle,
      strides = [1,1,1,1],padding='SAME'
    )
    
    pass_size = list(self.input.shape[1:])
    grad_pass = tf.nn.conv2d_backprop_input(
      input_sizes =[batch_size]+pass_size,
      filter = self.w,
      out_backprop = grad_middle,
      strides = [1,1,1,1],padding='SAME'
    )

    update_w = []
    tf.assign(self.m, 0.7*self.m + learning_rate*grad)    
    tf.assign(self.w,self.w - self.m)
    return grad_pass,update_w





# =========== Make Class  ===========
l1 = CNNLayer(kernel=7,in_c=3,out_c=256)
l2 = CNNLayer(kernel=1,in_c=256,out_c=256)

l3 = CNNLayer(kernel=5,in_c=256,out_c=256)
l4 = CNNLayer(kernel=1,in_c=256,out_c=256)

l5 = CNNLayer(kernel=3,in_c=256,out_c=256)
l6 = CNNLayer(kernel=1,in_c=256,out_c=256)

l7 = CNNLayer(kernel=2,in_c=256,out_c=256)
l8 = CNNLayer(kernel=1,in_c=256,out_c=256)

l9 = CNNLayer(kernel=2,in_c=256,out_c=256)
l10 = CNNLayer(kernel=1,in_c=256,out_c=10)

# === Hyper Param ===
num_epoch =  165000 
num_epoch =  10001 
learning_rate = 0.005
print_size = 100

batch_size = 100



# =========== Make Graph  ===========
x  = tf.placeholder(tf.float32, [None, 32, 32, 3],name="x-input")
y = tf.placeholder(tf.float32, [None, NUM_CLASSES],name="y-input")

layer1 = l1.feedforward(x)
layer2 = l2.feedforward_avg(layer1)

layer3 = l3.feedforward(layer2)
layer4 = l4.feedforward_avg(layer3)

layer5 = l5.feedforward(layer4)
layer6 = l6.feedforward_avg(layer5)

layer7 = l7.feedforward(layer6)
layer8 = l8.feedforward_avg(layer7)

layer9 = l9.feedforward(layer8)
print(layer9.shape,'===============')
layer10 = l10.feedforward_avg(layer9)

results = tf.reshape(layer10,(-1,NUM_CLASSES))

# ====== my total cost =====
# final_soft = tf_softmax(results)
# cost = tf.reduce_mean(-1.0 * (y* tf.log(final_soft) + (1-y)*tf.log(1-final_soft)))
# cost = tf.reduce_sum(-1.0 * (y* tf.log(final_soft) + (1-y)*tf.log(1-final_soft)))
# tf.add_to_collection('losses', cost)
# correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# ====== my total cost =====

# ===== Auto Train ====
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=results, labels=y, name='cross_entropy_per_example')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
tf.add_to_collection('losses', cross_entropy_mean)
total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
tf_learning_rate = tf.placeholder(tf.float32)
optimizer = tf.train.MomentumOptimizer(learning_rate=tf_learning_rate, momentum=0.9).minimize(total_loss)
# ===== Auto Train ====

#Predictions
tf_prediction = tf_softmax(results)

# === Back Propagation === 
grad_10,grad_10w = l10.backprop_avg(tf.reshape(tf_prediction-y,[batch_size,1,1,NUM_CLASSES]))
grad_9, grad_9w  = l9.backprop(grad_10)

grad_8, grad_8w  = l8.backprop_avg(grad_9)
grad_7, grad_7w  = l7.backprop(grad_8)

grad_6, grad_6w  = l6.backprop_avg(grad_7)
grad_5, grad_5w  = l5.backprop(grad_6)

grad_4, grad_4w  = l4.backprop_avg(grad_5)
grad_3, grad_3w  = l3.backprop(grad_4)

grad_2, grad_2w  = l2.backprop_avg(grad_3)
grad_1, grad_1w  = l1.backprop(grad_2)

weight_update = grad_10w+grad_9w+ \
                grad_8w+grad_7w+ \
                grad_6w+grad_5w+ \
                grad_4w+grad_3w+ \
                grad_2w+grad_1w



# =========== Session ===========
config = tf.ConfigProto(device_count = {'GPU': 1})
sess = tf.Session(config=config)
with sess:
    sess.run(tf.global_variables_initializer())

    step_per_epoch = int(len(train_labels)/batch_size)

    for step in range(num_epoch):

      #Shuffeling the data on each epoch - per each step
        if step % step_per_epoch == 0:
                shuffle_indices = np.random.permutation(np.arange(len(train_images)))
                train_images = train_images[shuffle_indices]
                train_labels = train_labels[shuffle_indices]
                # indices_list = create_indices(train_labels,NUM_CLASSES)
        if step == 9000: learning_rate = 0.001
        if step == 12000: learning_rate = 0.0003
        if step == 19000: learning_rate = 0.00005

        #Creating batch data
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_images[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        #Training
        feed_dict = {x: batch_data, y : batch_labels, tf_learning_rate: learning_rate}
        # _, loss_out, predictions = sess.run([optimizer, total_loss, tf_prediction], feed_dict=feed_dict)
        sess_result = sess.run([weight_update,optimizer, total_loss, tf_prediction], feed_dict=feed_dict)

        print(sess_result[0].shape)
        sys.exit()
        
        
        #Accuracy
        if (step % print_size == 0):
            print('\n===================')
            print('Minibatch loss at step %d: %f' % (step, loss_out))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            accuracy_final = 0.0
            test_predictions = np.ndarray(shape=(len(test_labels),NUM_CLASSES),dtype=np.float32)

            for i in range(20):
                offset = i*500
                feed_dict = {x: test_images[offset:(offset+500)], y : test_labels[offset:(offset+500)]}
                test_predictions[offset:(offset+500)] = sess.run(tf_prediction, feed_dict = feed_dict)
                accuracy_final+=accuracy(test_predictions[offset:(offset+500)], test_labels[offset:(offset+500)])
            print('elu network 80000 steps')
            print('Test accuracy is %.1f%%' %(accuracy_final/20))



# -- end code --