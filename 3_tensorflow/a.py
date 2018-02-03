import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

np.random.seed(65558)
tf.set_random_seed(65558)


def tf_log(x):
    return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
def d_tf_log(x):
    return tf.multiply(tf_log(x),tf.subtract(  tf.constant(1.0), tf_log(x))  )

def tf_arctan(x):
    return tf.atan(x)
def d_tf_arctan(x):
    return tf.div(tf.constant(1.0),tf.subtract( tf.constant(1.0) , tf.square(x) ))


# 0. Prepare the Training Data
train = mnist.test
images, labels = train.images, train.labels
only_zero_index,only_one_index = np.where(labels==0)[0],np.where(labels==1)[0]
only_zero_image,only_zero_label = images[[only_zero_index]],np.expand_dims(labels[[only_zero_index]],axis=1)
only_one_image,only_one_label   = images[[only_one_index]],np.expand_dims(labels[[only_one_index]],axis=1)

images = np.vstack((only_zero_image,only_one_image))
labels = np.vstack((only_zero_label,only_one_label))
images,label = shuffle(images,labels)

test_image_num = 10
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:,:],label[test_image_num:,:]

num_epoch = 5
total_cost = 0
graph = tf.Graph()




# 1.5 Declare Hyper Parameter on the Default Graph ---- HOW MANY DO WE NEED? WHAT WILL THIS EFFECT THE NETWORK 
with graph.as_default():

    lr = tf.Variable(tf.constant(0.01))
    w1 = tf.Variable(tf.random_normal([784,356]))
    w2 = tf.Variable(tf.random_normal([356,128]))
    w3 = tf.Variable(tf.random_normal([128,1]))













# 1. Make the TF Focus on Network arch  on the Default Graph
with graph.as_default():

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 1])

    l1 = tf.matmul(x,w1)
    l1A = tf_log(l1)

    l2 = tf.matmul(l1A,w2)
    l2A = tf_arctan(l2)

    l3 = tf.matmul(l2A,w3)
    l3A = tf_log(l3)

    cost = tf.multiply(tf.square(tf.subtract(l3A,y)),tf.constant(0.5))
    cost_auto = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    grad_3_part_1 = tf.subtract(l3A,y)
    grad_3_part_2 = d_tf_log(l3)
    grad_3_part_3 = l2A
    grad_3 = tf.matmul(tf.transpose(grad_3_part_3), tf.multiply(grad_3_part_1,grad_3_part_2) )

    grad_2_part_1 = tf.matmul( tf.multiply(grad_3_part_1,grad_3_part_2),  tf.transpose(w3) )
    grad_2_part_2 = d_tf_arctan(l2)
    grad_2_part_3 = l1A
    grad_2 = tf.matmul(tf.transpose(grad_2_part_3), tf.multiply(grad_2_part_1,grad_2_part_2) )

    grad_1_part_1 = tf.matmul( tf.multiply(grad_2_part_1,grad_2_part_2),  tf.transpose(w2))
    grad_1_part_2 = d_tf_log(l1)
    grad_1_part_3 = x
    grad_1 = tf.matmul(tf.transpose(grad_1_part_3), tf.multiply(grad_1_part_1,grad_1_part_2) )

    update = [
        tf.assign(w3,tf.subtract( w3, tf.multiply(lr,grad_3) ) ),
        tf.assign(w2,tf.subtract( w2, tf.multiply(lr,grad_2) ) ),
        tf.assign(w1,tf.subtract( w1, tf.multiply(lr,grad_1) ) )
    ]
    







# 2. Make the TF session
with tf.Session(graph=graph,config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        for current_image_index in range(len(training_images)):
            
            current_image = np.expand_dims(training_images[current_image_index],axis=0)
            current_index = np.expand_dims(training_lables[current_image_index],axis=0)
            
            output = sess.run([cost,update],feed_dict={x:current_image,y:current_index})
            total_cost = total_cost + output[0]
        print("Current Iter: ", iter, " current cost: ", total_cost)
        total_cost = 0

    for current_image_index in range(len(testing_images)):
        current_image = np.expand_dims(testing_images[current_image_index],axis=0)
        output = sess.run([l3A],feed_dict={x:current_image})
        print(np.round(output[0]), ' : ',current_index)
# -- end code --