import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from six.moves import cPickle as pickle
from read_10_data import get_data

np.random.seed(789)
tf.set_random_seed(789)


#Accuracy measurment for the results
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# get data
NUM_CLASSES = 10
train_images, train_labels, test_images,test_labels = get_data()

# === Hyper Param ===
num_epoch =  165000 
num_epoch =  10001 
learning_rate = 0.01
print_size = 100

batch_size = 100

proportion_rate = 1000
decay_rate = 0.08


# =========== Session ===========
config = tf.ConfigProto(device_count = {'GPU': 1})
sess = tf.Session(config=config)
with sess:
    sess.run(tf.global_variables_initializer())

    step_per_epoch = int(len(train_labels)/batch_size)

    print("==========================")
    print('Number of Training Images: ',len(train_labels))
    print('Steps per epoch: ',step_per_epoch)

    for step in range(num_epoch):

      #Shuffeling the data on each epoch - per each step
        if step % step_per_epoch == 0:
                shuffle_indices = np.random.permutation(np.arange(len(train_images)))
                train_images = train_images[shuffle_indices]
                train_labels = train_labels[shuffle_indices]
                # indices_list = create_indices(train_labels,NUM_CLASSES)
                print(step)
                input()

        if step == 9000: learning_rate = 0.001
        if step == 12000: learning_rate = 0.0003
        if step == 19000: learning_rate = 0.00005

        #Creating batch data
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_images[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        #Training
        # feed_dict = {x: batch_data, y : batch_labels, tf_learning_rate: learning_rate}
        # _, loss_out, predictions = sess.run([optimizer, total_loss, tf_prediction], feed_dict=feed_dict)
        # sess_result = sess.run([weight_update, total_loss, tf_prediction],  feed_dict={x: batch_data, y : batch_labels, tf_learning_rate: learning_rate,iter_variable_dil:step})

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