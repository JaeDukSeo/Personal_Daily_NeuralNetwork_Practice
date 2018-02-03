import tensorflow as tf
import numpy as np

# 1. Preprocess Data 
# 2. Make graph 
# 2.1 input 
# 2.2 operations
# 2.3 cost func
# 3. Make Session
# 3.1 variable
# 3.2 cal

# 1. Preprocess data 
trX = np.linspace(-1, 1, 101)
trY = 4 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise

# 2. 
graph = tf.Graph()
with graph.as_default():

    X = tf.placeholder("float",name="input_X") # create symbolic variables
    Y = tf.placeholder("float",name="input_y")

    w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
    y_predicted_model = tf.multiply(X, w)

    cost = tf.square(Y - y_predicted_model) # use square error for cost function

    # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    # train_op = tf.train.AdadeltaOptimizer(0.1).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    # train_op = tf.train.MomentumOptimizer(0.001).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    # train_op = tf.train.FtrlOptimizer(0.001).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
    train_op = tf.train.RMSPropOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

    #  Funny finding - not all of the optimizers are fitted for this taks
    #  the initializing value matters alot!
    #  Interesting - the variable is cloer but the error rate is higher -> maybe the square....

# 3. 
with tf.Session(graph =  graph) as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(100):
        for (x, y) in zip(trX, trY):
            printing = sess.run([cost,train_op], feed_dict={X: x, Y: y})
        if i % 10 == 0:
            print(printing)
            print(sess.run(w))

    print(sess.run(w)) # It should be something around 2











# -------- END OF THE CODE -------