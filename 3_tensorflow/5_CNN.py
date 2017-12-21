import os,numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf,numpy as np,time
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt


# 1. Data preprocess
# 2. Make the graph
# 3. Make the session

# Func: Show the image data
def show_minst_data(teX):
    plt.imshow(np.squeeze(teX),cmap='gray')
    plt.show()

# 1. Data process - batch processing
batch_size = 128
test_size = 256

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img


# 2. Mkae the graph
graph  = tf.Graph()
with graph.as_default():

    # 2.1 Inputted Data
    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 10])
    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    # 2.2 make the opertions
    w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    l1a = tf.nn.elu(tf.nn.conv2d(X, w1,strides=[1, 1, 1, 1], padding='SAME'))# l1a shape=(?, 28, 28, 32)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')# l1 shape=(?, 14, 14, 32)
    l1 = tf.nn.dropout(l1, p_keep_conv)

    w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    l2a = tf.nn.elu(tf.nn.conv2d(l1, w2,strides=[1, 1, 1, 1], padding='SAME'))  # l2a shape=(?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # l2 shape=(?, 7, 7, 64)
    l2 = tf.nn.dropout(l2, p_keep_conv)

    # w3= tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    # l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,strides=[1, 1, 1, 1], padding='SAME'))  # l3a shape=(?, 7, 7, 128)
    # l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # l3 shape=(?, 4, 4, 128)
    # l3 = tf.nn.dropout(l3, p_keep_conv)

    w4 = tf.Variable(tf.random_normal([64 * 7 * 7,625], stddev=0.01))
    l4 = tf.reshape(l2, [-1, w4.get_shape().as_list()[0]  ])    # reshape to (?, 2048)
    l4 = tf.matmul(l4, w4)
    l4 = tf.nn.elu(l4)
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    wo = tf.Variable(tf.random_normal([625,10], stddev=0.01))
    final = tf.matmul(l4, wo)

    # 2.3 make the cost functions
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=final, labels=Y)
    cost = tf.reduce_mean(cost)
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(final, 1)


# 3. Make the session
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        test_result = sess.run(predict_op, feed_dict={X: teX[test_indices], p_keep_conv: 1.0,p_keep_hidden: 1.0})
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==  test_result ))


# ---- END OF TE CODE -----
