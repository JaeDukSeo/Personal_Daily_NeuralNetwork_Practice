import tensorflow as tf
import numpy as np
import os
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data
# 0. Preprocess the data
# 1, Make the graph
#   1.1 Make the place holder
#   1.3 Make the operations
#   1.5 Make the cost function
# 2. Make the session
#   2.1 Make the variables
#   2.3 Make the loop



# 0. Read the images and preprocess the class
PathDicom = ["../CPS_40_Data/1_Normal_I/resized/","../CPS_40_Data/2_adenoma_II/resized/","../CPS_40_Data/3_Cancer_III/resized/"]
normal_data,adenoma_data, cancer_data = [],[],[]
all_data = [normal_data,adenoma_data,cancer_data]
for path in range(len(PathDicom)):
    for dirName, subdirList, fileList in os.walk(PathDicom[path]):
        for filename in fileList:
            if ".jpg" in filename.lower():  # check whether the file's DICOM
                all_data[path].append(misc.imread(os.path.join(dirName,filename)))
    
noraml_train,normal_test = all_data[0][:-10],   all_data[0][-10:]
adenoma_train,adenoma_test = all_data[1][:-10], all_data[1][-10:]
cancer_train,cancer_test = all_data[2][:-10],   all_data[2][-10:]

noraml_train_label,normal_test_label =      np.ones([len(noraml_train),3]) * [1,0,0],     np.ones([len(normal_test),3]) * [1,0,0]
adenoma_train_label,adenoma_test_label =    np.ones([len(adenoma_train),3]) * [0,1,0],    np.ones([len(adenoma_test),3]) * [0,1,0]
cancer_train_label,cancer_test_label =      np.ones([len(cancer_train),3]) * [0,0,1],     np.ones([len(cancer_test),3]) * [0,0,1]



training_data =         np.vstack((noraml_train,adenoma_train,cancer_train))
training_data_label = np.vstack((noraml_train_label,adenoma_train_label,cancer_train_label))

test_data = np.vstack((normal_test,adenoma_test,cancer_test))
test_data_label = np.vstack((normal_test_label,adenoma_test_label,cancer_test_label))


# 1. Make the graph
graph = tf.Graph()
with graph.as_default():

    # 1.1 Declare the images first and the class
    X = tf.placeholder("float", [None, 450, 450, 3])
    Y = tf.placeholder("float", [None, 3])

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    w   = tf.Variable(tf.random_normal([3,3,3,32], stddev=0.01))                # 3x3x1 conv, 32 outputs
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l1  = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  strides=[1, 2, 2, 1], padding='SAME')  # l1 shape=(?, 14, 14, 32)
    l1  = tf.nn.dropout(l1, p_keep_conv)

    # w2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))   # 3x3x32 conv, 64 outputs
    # l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    # l2  = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  strides=[1, 2, 2, 1], padding='SAME')  # l1 shape=(?, 14, 14, 32)
    # l2  = tf.nn.dropout(l2, p_keep_conv)

    w3 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))   # 3x3x32 conv, 128 outputs
    l3a = tf.nn.relu(tf.nn.conv2d(l1, w3, strides=[1, 1, 1, 1], padding='SAME'))
    
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, 817216])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    w4 = tf.Variable(tf.random_normal([817216,625], stddev=0.01))# FC 128 * 4 * 4 inputs, 625 outputs
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    w_o = tf.Variable(tf.random_normal([625,3], stddev=0.01))       # FC 625 inputs, 10 outputs (labels)
    pyx = tf.matmul(l4, w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pyx, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(pyx, 1)

# 2. Make the Session
with tf.Session(graph = graph) as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 1
    test_size = 30

    for i in range(100):
        training_batch = zip(range(0, len(training_data), batch_size), range(batch_size, len(training_data)+1, batch_size))

        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: training_data[start:end], 
                                          Y: training_data_label[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(test_data)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(test_data_label[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: test_data[test_indices],
                                                                                                 p_keep_conv: 1.0, p_keep_hidden: 1.0})))







# ---------- END OF THE CODE ------