
import numpy as np,sys
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(678)
tf.set_random_seed(678)

def tf_log(x):
    return tf.div(tf.constant(1.0),tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
def d_tf_log(x):
    return tf.multiply(tf_log(x),tf.subtract(  tf.constant(1.0), tf_log(x))  )

def tf_tanh(x):
    return tf.tanh(x)
def d_tf_tanh(x):
    return tf.subtract(tf.constant(1.0),tf.square(tf.tanh(x)))

def tf_arctan(x):
    return tf.atan(x)
def d_tf_arctan(x):
    return tf.div(tf.constant(1.0),tf.subtract( tf.constant(1.0) , tf.square(x) ))


# 0. Declare Training Data and Labels
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

train = mnist.test
images, labels = train.images, train.labels
only_zero_index,only_one_index = np.where(labels==0)[0],np.where(labels==1)[0]
only_zero_image,only_zero_label = images[[only_zero_index]],np.expand_dims(labels[[only_zero_index]],axis=1)
only_one_image,only_one_label   = images[[only_one_index]],np.expand_dims(labels[[only_one_index]],axis=1)

images = np.vstack((only_zero_image,only_one_image))
labels = np.vstack((only_zero_label,only_one_label))
images,label = shuffle(images,labels)

test_image_num,training_image_num = 20,100
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]

num_epoch = 100
total_cost = 0
cost_array =[]
graph = tf.Graph()














# 1. Think What weights do I need? And how to initialize them 
with graph.as_default():

    lr_x = tf.Variable(tf.constant(0.001))
    lr_rec = tf.Variable(tf.constant(0.000001))
    lr_sg = tf.Variable(tf.constant(0.0001))
    
    hidden_states = tf.Variable(tf.random_normal([784,3]))

    w_x = tf.Variable(tf.random_normal([784,784],stddev=0.45) *  tf.constant(0.2)) 
    w_rec = tf.Variable(tf.random_normal([784,784],stddev=0.035) *  tf.constant(0.2))
    w_fc = tf.Variable(tf.random_normal([784,1],stddev=0.95)*  tf.constant(0.2))

    w_sg_1 = tf.Variable(tf.random_normal([784,784],stddev=0.35)*  tf.constant(0.2))
    w_sg_2 = tf.Variable(tf.random_normal([784,784],stddev=0.35)*  tf.constant(0.2))
















# 2. Think What is the structure of this Network? How does it work - EVEN BACK PROP!
with graph.as_default():

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 1])
    update = []
    h_update = []

    l1 = tf.add(tf.matmul(x,w_x),tf.matmul(tf.expand_dims(hidden_states[:,0],axis=0), w_rec  )  )
    l1A = tf_tanh(l1)  
    h_update.append(tf.assign(hidden_states[:,1],tf.squeeze(l1A) ))

    # # ----- Time Stamp 1 Syn Grad Update ------
    grad_1sg_part_1 = tf.matmul(l1A,w_sg_1)
    grad_1sg_part_2 = d_tf_tanh(l1)
    grad_1sg_part_rec = tf.expand_dims(hidden_states[:,0],axis=0)
    grad_1sg_part_x = x
    
    grad_1sg_rec = tf.matmul(tf.transpose(grad_1sg_part_rec),tf.multiply( grad_1sg_part_1,grad_1sg_part_2 ) )
    grad_1sg_x =   tf.matmul(tf.transpose(grad_1sg_part_x),  tf.multiply( grad_1sg_part_1,grad_1sg_part_2 ) )
    
    update.append(tf.assign(w_rec, tf.add(w_rec, tf.multiply(lr_rec,grad_1sg_rec)))  )
    update.append(tf.assign(w_x,   tf.add(w_x,   tf.multiply(lr_rec,grad_1sg_x)))   )
    grad_true_0 = tf.matmul(tf.multiply(grad_1sg_part_1,grad_1sg_part_2),tf.transpose(w_rec))
    # # ----- Time Stamp 1 Syn Grad Update ------

    l2 = tf.add(tf.matmul(x,w_x),tf.matmul(tf.expand_dims(hidden_states[:,1],axis=0),w_rec)    )  
    l2A = tf_tanh(l2)  
    h_update.append(tf.assign(hidden_states[:,2], tf.squeeze(l2A) ))
     
    # # ----- Time Stamp 2 Syn Grad Update ------
    grad_2sg_part_1 = tf.matmul(l2A,w_sg_2)
    grad_2sg_part_2 = d_tf_tanh(l2)
    grad_2sg_part_rec = tf.expand_dims(hidden_states[:,1],axis=0)
    grad_2sg_part_x =  x
    
    grad_2sg_rec = tf.matmul(tf.transpose(grad_2sg_part_rec),tf.multiply( grad_2sg_part_1,grad_2sg_part_2 ) )
    grad_2sg_x =   tf.matmul(tf.transpose(grad_2sg_part_x),  tf.multiply( grad_2sg_part_1,grad_2sg_part_2 ) )
    
    update.append(tf.assign(w_rec, tf.add(w_rec, tf.multiply(lr_rec,grad_2sg_rec)))   )
    update.append(tf.assign(w_x,   tf.add(w_x,   tf.multiply(lr_rec,grad_2sg_x)))   )
    grad_true_1_from_2 = tf.matmul(tf.multiply(grad_2sg_part_1,grad_2sg_part_2),tf.transpose(w_rec))
    # # ----- Time Stamp 2 Syn Grad Update ------

    # # ----- Time Stamp 1 True Gradient Update ------
    grad_true_1_part_1 = tf.subtract(grad_1sg_part_1,grad_true_1_from_2)
    grad_true_1_part_2 = tf.expand_dims(hidden_states[:,1],axis=0)
    grad_true_1 = tf.matmul(tf.transpose(grad_true_1_part_2),grad_true_1_part_1)
    update.append(tf.assign(w_sg_1,tf.subtract(w_sg_1,tf.multiply(lr_sg,grad_true_1))))
    # # ----- Time Stamp 1 True Gradient Update ------

    # # ----- Fully Connected for Classification ------
    l3 = tf.matmul( tf.expand_dims(hidden_states[:,2],axis=0),w_fc)
    l3A = tf_log(l3)
    # # ----------------------------------------------

    # # -- MAN BACK PROP -----
    cost = tf.multiply(tf.square(tf.subtract(l3A,y)),tf.constant(0.5))
    # # -- MAN BACK PROP -----
    
    # # -- AUTO BACK PROP -----
    cost_auto = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    # # -- AUTO BACK PROP -----

    # # ------- FC weight update ---------------------
    grad_fc_part_1 = tf.subtract(l3A,y)
    grad_fc_part_2 = d_tf_log(l3)
    grad_fc_part_3 = tf.expand_dims(hidden_states[:,2],axis=0)
    grad_fc = tf.matmul(tf.transpose(grad_fc_part_3),tf.multiply(grad_fc_part_1,grad_fc_part_2))
    update.append(tf.assign(w_fc,tf.subtract(w_fc,tf.multiply(lr_x,grad_fc))))

    grad_true_2_from_3 = tf.matmul(tf.multiply(grad_fc_part_1,grad_fc_part_2),tf.transpose(w_fc))
    # # ------- FC weight update ---------------------
        
    # # ----- Time Stamp 2 True Gradient Update ------
    grad_true_2_part_1 = tf.subtract(grad_2sg_part_1,grad_true_2_from_3)
    grad_true_2_part_2 = tf.expand_dims(hidden_states[:,2],axis=0)
    grad_true_2 = tf.matmul(tf.transpose(grad_true_2_part_2),grad_true_2_part_1)
    update.append(tf.assign(w_sg_2,tf.subtract(w_sg_2,tf.multiply(lr_sg,grad_true_2))) ) 
    # # ----- Time Stamp 2 True Gradient Update ------


























# 3. Run the Session, however think about how many batches etc
with tf.Session(graph=graph) as sess:

    sess.run(tf.global_variables_initializer())
    total_cost = 0 

    for iter in range(num_epoch):
        for current_image_index in range(len(training_images)):
            
            current_image = np.expand_dims(training_images[current_image_index],axis=0)
            current_index = np.expand_dims(training_lables[current_image_index],axis=0)
            
            # If you wish to do manual back prop
            output = sess.run([cost,update,h_update],feed_dict={x:current_image,y:current_index})

            # If you wish to use the auto differential
            # output = sess.run([cost,cost_auto,h_update],feed_dict={x:current_image,y:current_index})

            total_cost = total_cost + output[0].sum()
        print("Current Iter: ", iter, " current cost: ", total_cost)
        cost_array.append(total_cost)
        total_cost = 0

    
    plt.plot(np.arange(num_epoch), cost_array)
    plt.show()


    for current_image_index in range(len(testing_images)):
        current_image = np.expand_dims(testing_images[current_image_index],axis=0)
        current_label = testing_lables[current_image_index]
        output = sess.run([l3A,h_update],feed_dict={x:current_image})
        print(current_image_index,' : ' ,output[0], ' : ',np.round(output[0])," : ",current_label)












# ---- end code ---
