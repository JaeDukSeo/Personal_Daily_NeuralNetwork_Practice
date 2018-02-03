import numpy as np
import tensorflow as tf
from sklearn import datasets
import sys,os,numpy as np,sklearn
from scipy import misc
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
  W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
  conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
  if with_bias:
    return conv + bias_variable([ out_features ])
  return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
  current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = tf.nn.relu(current)
  current = conv2d(current, in_features, out_features, kernel_size)
  current = tf.nn.dropout(current, keep_prob)
  return current

def block(input, layers, in_features, growth, is_training, keep_prob):
  current = input
  features = in_features
  for idx in range(int(layers)):
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
    current = tf.concat( (current, tmp),3)
    features += growth
  return current, features

def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')


graph = tf.Graph()
weight_decay = 1e-4
with graph.as_default():

    tf_train_dataset = tf.placeholder('float',[None,28,28,1],name='xs')
    ys = tf.placeholder('float',[None,10],name='ys')
    lr = tf.placeholder("float", shape=[],name="lr")
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    is_training = tf.placeholder("bool", shape=[],name="is_training")
    mini_batch = tf.placeholder(tf.int32,name='mini_batch')

    current = conv2d(tf_train_dataset, 1, 10, 3)

    current, features = block(current, 3, 10, 8, is_training, keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    current = avg_pool(current, 2)

    current2, features2 = block(current, 3, features, 8, is_training, keep_prob)
    current2 = batch_activ_conv(current2, features2, features2, 1, is_training, keep_prob)
    current2 = avg_pool(current2, 9)

    current3 = tf.reshape(current2, [ mini_batch, -1 ])

    w_full_1 = tf.Variable(tf.random_normal([features2,40],stddev=0.01))
    b_full_1 = bias_variable([ 40 ])
    current_4 = tf.matmul(current3,w_full_1)
    current_4 = current_4 + b_full_1
    current_4 = tf.nn.relu(current_4,name='current_4')
    current_4 = tf.nn.dropout(current_4, keep_prob)

    w_full_2 = tf.Variable(tf.random_normal([40,10],stddev=0.01))
    b_full_2 = bias_variable([ 10 ])
    current_5 = tf.matmul(current_4,w_full_2)
    current_5 = current_5 + b_full_2

    ys_ = tf.nn.softmax(current_5 ,name="predicted_value")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ys_, labels=ys),name='loss_value')
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)    

config = tf.ConfigProto(device_count = {'GPU': 0})
with tf.Session(graph=graph,config=config) as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    for iter in range(100000):
        current_x,current_label = sklearn.utils.shuffle(trX,trY)

        for i in range(0,len(trX),10):
            current_test = current_x[i:i+10]
            current_test_label = current_label[i:i+10]
            temp_data = session.run([optimizer,loss],feed_dict={tf_train_dataset:current_test,
                                                ys: current_test_label,mini_batch:10,
                                                lr:0.0001, keep_prob:0.8,
                                                is_training:True} )

            # print("Epoch: ", iter,' Loss: ', temp_data[1],'  Batch : ',i,' ',i+10)

# /Users/jaedukseo/Desktop/personal_cps_40_aov/4_tensorflow/z_temp_image_data
        saver.save(session, './z_temp_image_data/')
        for j in range(0,10,10):

            current_test = teX[j:j+10]
            current_test_label = teY[j:j+10]

            test_data = session.run([ys_,optimizer,loss],feed_dict={tf_train_dataset:current_test,
                                                        ys: current_test_label,mini_batch:10,
                                                        lr:0.0001, keep_prob:1.0,is_training:False} )
            
            print "Predicted: ",test_data[0]
            print "GT : ",current_test_label
            print("\nTEST DATA Epoch: ", iter,' Loss: ', test_data[1],'\n')





