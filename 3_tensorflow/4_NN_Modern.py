import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf,numpy as np,time
from tensorflow.examples.tutorials.mnist import input_data


# 1. Read the data and set the training and test
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# 2. Make the graph 
graph = tf.Graph()
with graph.as_default():

    # 2.1 input and output
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])
    p_keep_hidden = tf.placeholder("float")

    # 2.2 Hidden and output layer
    w_1 = tf.Variable(tf.random_normal([784, 625], stddev=0.1),name="w_1") # create symbolic variables
    X = tf.nn.dropout(X, p_keep_hidden)
    layer_one = tf.matmul(X,w_1)
    layer_one = tf.nn.relu(layer_one)

    w_2 = tf.Variable(tf.random_normal([625, 625], stddev=0.1),name="w_2") # create symbolic variables
    layer_one = tf.nn.dropout(layer_one, p_keep_hidden)
    layer_two = tf.matmul(layer_one,w_2)
    layer_two = tf.nn.relu(layer_two)

    w_o = tf.Variable(tf.random_normal([625, 10], stddev=0.1),name="w_o")
    layer_two = tf.nn.dropout(layer_two, p_keep_hidden)
    final_layer = tf.matmul(layer_two,w_o)

    # 2.3 the cost function and the variable - cost function is the reduced mean value
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_layer, labels=Y)) # compute mean cross entropy (softmax is applied internally)
    
    # 2.4 Optmizer method and algos
    # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct optimizer
    # train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    # train_op = tf.train.AdadeltaOptimizer(0.1).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    # train_op = tf.train.MomentumOptimizer(0.001).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    # train_op = tf.train.FtrlOptimizer(0.001).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    train_op = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

    predict_op = tf.argmax(final_layer, 1) 

# 3. Make the session
with tf.Session(graph=graph) as sess:
    # you need to initialize all variables
    sess.run(tf.global_variables_initializer())

    for i in range(100):

        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            result = sess.run([layer_one,train_op,cost], feed_dict= {X: trX[start:end], Y: trY[start:end] , p_keep_hidden:0.6 })
            # print(result[2])
        
        print(i, np.mean(np.argmax(teY, axis=1) ==  sess.run(predict_op, feed_dict={X: teX,p_keep_hidden:1.0})))
        # time.sleep(0.5)
        # next_step = raw_input('Press enter')


# This network will not perform more well then the other ones 
# since it have less deminsion to carry the operations 
# as well as other stuff have to be there in order to make the operations 
# more successfull.  - and moderen net


# With the modern architecture - the results are better however it takes much more time to 
# process as well as - there is a huge time aspect that we need to 
# take into consideration of










# -------- END OF THE CODE -----