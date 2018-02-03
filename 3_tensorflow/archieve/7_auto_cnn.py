import tensorflow as tf 
import numpy as np,sys
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 0. Data preprocess
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_data, x_label, y_data, y_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = y_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img



# 1. make graph
graph = tf.Graph()
with graph.as_default():

    x = tf.placeholder('float',[None,28,28,1])

    w1 = tf.Variable(tf.random_normal([4, 4, 1, 10], stddev=0.01))
    w2 = tf.Variable(tf.random_normal([10* 28 * 28,635], stddev=0.01))
    w3 = tf.Variable(tf.random_normal([635,28*28*10], stddev=0.01))
    w4 = tf.Variable(tf.random_normal([3,3,10,1], stddev=0.01))

    layer_1 = tf.nn.conv2d(x,w1,strides=[1, 1, 1, 1], padding='SAME')
    layer_1_act = tf.nn.sigmoid(layer_1)

    layer_2_data = tf.reshape(layer_1_act,[-1,w2.get_shape().as_list()[0] ])
    layer_2 = tf.matmul(layer_2_data,w2)
    layer_2_act = tf.nn.sigmoid(layer_2)

    layer_3 = tf.matmul(layer_2_act,w3)
    layer_3_act = tf.nn.sigmoid(layer_3)

    final_data = tf.reshape(layer_3_act,[-1,28,28,10])
    final = tf.nn.conv2d(final_data,w4,strides=[1,1,1,1],padding='SAME')

    loss = tf.reduce_mean(tf.pow(final - x, 2))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss )


# 2. Make Session
sess = tf.Session(graph = graph)
past_i = 0
with sess: 

    sess.run(tf.global_variables_initializer())

    for iter in range(100):

        for i in range(1000,56000,1000):

            current_x_batch = trX[past_i:i]
            current_result,current_loss = sess.run([optimizer,loss],feed_dict={  x:current_x_batch   })

            print "Current Epoch: ", iter, "  Current Loss: ", current_loss," Current mini Batch : ",past_i,'    :  ',i

            past_i= i

        past_i = 0 

        # ------- TESTING CODE USED -------------
        n = 4
        canvas_orig = np.empty((28 * n, 28 * n))
        canvas_recon = np.empty((28 * n, 28 * n))
        for i in range(n):
            # MNIST test set
            batch_x, _ = mnist.test.next_batch(n)
            batch_x = batch_x.reshape(-1, 28, 28, 1)
            # Encode and decode the digit image
            g = sess.run(final, feed_dict={x: batch_x})
            
            # Display original images
            for j in range(n):
                # Draw the generated digits
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
            # Display reconstructed images
            for j in range(n):
                # Draw the generated digits
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

        print("Original Images")     
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.show()

        print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()




# ------------ END OF THE CODE ----