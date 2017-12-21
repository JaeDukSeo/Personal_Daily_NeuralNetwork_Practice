import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np,time


# 1. preprocess the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


# 2. Make the graph 
graph = tf.Graph()
with graph.as_default():

    # ** NOTE ** batch normalize so we are not going to have that shape
    X = tf.placeholder("float", [None, 784],name='input_x') # create symbolic variables
    Y = tf.placeholder("float", [None, 10], name='input_y')

    # 2.1 Weigth and the operations
    w_1 =  tf.Variable(tf.random_normal([784, 120], stddev=0.01),name="w_1")
    layer_one = tf.matmul(X, w_1)
    layer_one = tf.nn.sigmoid(layer_one)

    w_2 =  tf.Variable(tf.random_normal([120, 300], stddev=0.1),name="w_2")
    layer_two = tf.matmul(layer_one, w_2)
    layer_two = tf.nn.sigmoid(layer_two)

    # ** NOTE ** The sigmoid - in many ways the activation functions - plays a huge roel 
    # and this needs to be more and more used in alot of different ways
    # non linearlyity very important VERY VERY FUCKING GOOD

    w_3 =  tf.Variable(tf.random_normal([300, 10], stddev=0.05),name="w_3")
    final_result = tf.matmul(layer_two, w_3)
    
    # 2.3 the cost function and the variable - cost function is the reduced mean value
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_result, labels=Y)) # compute mean cross entropy (softmax is applied internally)
    
    # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct optimizer
    # train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    # train_op = tf.train.AdadeltaOptimizer(0.1).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    # train_op = tf.train.MomentumOptimizer(0.001).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    # train_op = tf.train.FtrlOptimizer(0.001).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    train_op = tf.train.RMSPropOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

    predict_op = tf.argmax(final_result, 1) # at predict time, evaluate the argmax of the logistic regression

# 3. Make the session 
with tf.Session(graph=graph) as sess:

    # you need to initialize all variables
    sess.run(tf.global_variables_initializer())

    for i in range(100):

        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            result = sess.run([layer_one,train_op,cost], feed_dict= {X: trX[start:end], Y: trY[start:end]})
            # print(result[2])
        
        print(i, np.mean(np.argmax(teY, axis=1) ==  sess.run(predict_op, feed_dict={X: teX})))
        # time.sleep(0.5)
        # next_step = raw_input('Press enter')














# ---------- END OF THE CODE ---------