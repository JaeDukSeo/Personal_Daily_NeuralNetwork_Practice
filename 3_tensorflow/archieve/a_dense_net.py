import tensorflow as tf
import numpy as np,sys,sklearn
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 0. Data preprocess
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_data, x_label, y_data, y_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = y_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img

# 1. Make the graph 
graph = tf.Graph()
with graph.as_default():

    x = tf.placeholder('float',[None,28,28,1])

    w1 = tf.Variable(tf.random_normal([3,3,1,3],stddev=0.01))
    w2 = tf.Variable(tf.random_normal([3,3,1,3],stddev=0.01))


# 2. Make the Session
with tf.Session(graph = graph) as sess:

    sess.run(tf.global_variables_initializer())


sys.exit()
# ----------------ORIGINAL DENSE NET -----------------
def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):                              
  res = [ 0 ] * len(tensors)                                                                                           
  batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]                    
  total_size = len(batch_tensors[0][1])                                                                                
  batch_count = (total_size + batch_size - 1) / batch_size                                                             
  for batch_idx in xrange(batch_count):                                                                                
    current_batch_size = None                                                                                          
    for (placeholder, tensor) in batch_tensors:                                                                        
      batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                                         
      current_batch_size = len(batch_tensor)                                                                           
      feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                               
    tmp = session.run(tensors, feed_dict=feed_dict)                                                                    
    res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]                                                   
  return [ r / float(total_size) for r in res ]

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

# ------- image - depth - Feature Map - ??  -   traing ---- Drop out
def block(input, layers, in_features, growth, is_training, keep_prob):
  current = input
  features = in_features

  for idx in xrange(layers):
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
    current = tf.concat((current, tmp),3)

    # Growth Rate is 8 - An Intiger...?
    features =  features + growth

  return current, features

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
    # current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)

    return current
# ---------------------------------------------------------

def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

# 0. Process Data
# data_dir = 'data'
# image_size = 32
# image_dim = image_size * image_size * 3
# meta = unpickle(data_dir + '/batches.meta')
# label_names = meta['label_names']
# label_count = len(label_names)

# train_files = [ 'data_batch_%d' % d for d in xrange(1, 6) ]
# train_data, train_labels = load_data(train_files, data_dir, label_count)
# pi = np.random.permutation(len(train_data))
# train_data, train_labels = train_data[pi], train_labels[pi]
# test_data, test_labels = load_data([ 'test_batch' ], data_dir, label_count)
# print "Train:", np.shape(train_data), np.shape(train_labels)
# print "Test:", np.shape(test_data), np.shape(test_labels)

# data = { 'train_data': train_data,
#     'train_labels': train_labels,
#     'test_data': test_data,
#     'test_labels': test_labels }
# run_model(data, image_dim, label_count, 40)

label_count = 10
depth = 10
batch_size = 1000
number_of_epoch = 200
learning_rate = 0.1

# 1.0 Make the Grpah
weight_decay = 1e-4
layers = (depth - 4) / 3
graph = tf.Graph()
with graph.as_default():
    xs = tf.placeholder("float", shape=[None, 28,28,1])
    ys = tf.placeholder("float", shape=[None, 10])
    lr = tf.placeholder("float", shape=[])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder("bool", shape=[])

    current = tf.reshape(xs, [ -1, 28, 28, 1 ])
    current = conv2d(current, 1, 10, 3)

    current, features = block(current, layers, 10, 8, is_training, keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    current = avg_pool(current, 2)

    current, features = block(current, layers, features, 8, is_training, keep_prob)
    current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
    current = avg_pool(current, 2)

    current, features = block(current, layers, features, 4, is_training, keep_prob)
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current_f = avg_pool(current, 4)

    final_dim = features
    current_ff = tf.reshape(current_f, [ -1, final_dim ])
    Wfc,bfc = weight_variable([ final_dim, label_count ]),bias_variable([ label_count ])
    ys_ = tf.nn.softmax( tf.matmul(current_ff, Wfc) + bfc )

    cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
    correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 2. Make the Session
with tf.Session(graph=graph) as session:

    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print ' ---------------- Started Training ----------------' 

    # 2.7 Run the Epoch
    for iter in range(number_of_epoch):

        # 2.8 Make the Batch
        for i in range(0,len(x_label),1000):
            current_x = trX[i:i+1000]
            current_label = x_label[i:i+1000]
            batch_res = session.run([ train_step, cross_entropy, accuracy ],feed_dict = { xs: current_x, ys: current_label, r: learning_rate, is_training: True, keep_prob: 0.8 })
            print "Epoch: ", iter, " Accuracy : ", batch_res[2], '  Batch :', i , '  ', i + 1000

        # 3. Step Wise Learning Rate
        if iter == 150: 
            learning_rate = 0.01

        test_run = session.run([ ys_,cross_entropy, accuracy ],feed_dict = { xs: teX[:20], ys: y_label[:20], lr: learning_rate, is_training: False, keep_prob: 1.0 })
        print '\n\n-----------------------------'
        print "Epoch: ", iter, " Predicted : ", test_run[0],' Ground Truth : ',y_label[:20], ' Accuracy  : ',test_run[2]
        print '-----------------------------\n\n'
        

        # if batch_idx % 100 == 0: 
        #     print epoch, batch_idx, batch_res[1:]
        # save_path = saver.save(session, 'densenet_%d.ckpt' % epoch)
        # test_results = run_in_batch_avg(session, [ cross_entropy, accuracy ], [ xs, ys ],feed_dict = { xs: data['test_data'], ys: data['test_labels'], is_training: False, keep_prob: 1. })
        # print epoch, batch_res[1:], test_results


















# ------ END CODE --------