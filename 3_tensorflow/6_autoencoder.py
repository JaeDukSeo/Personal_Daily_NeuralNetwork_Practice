import tensorflow as tf
import numpy as np,sys
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 0. Data Preprocess
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_data, x_label, y_data, y_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = x_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = y_data.reshape(-1, 28, 28, 1)  # 28x28x1 input img

# 0.5 Declare Hyper parameters
learning_rate = 0.04
num_steps = 3
num_input,num_hidden_1,num_hidden_2 = 784,250,100

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=3, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


# 1. Make GRaph 
graph  = tf.Graph()
with graph.as_default():

    x = tf.placeholder("float",[None,num_input])

    # Endocers
    w1 = tf.Variable(tf.random_normal([num_input, num_hidden_1]))
    b1 = tf.Variable(tf.random_normal([num_hidden_1]))
    w2 = tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2]))
    b2 = tf.Variable(tf.random_normal([num_hidden_2]))
    
    # Decoders
    w3 =  tf.Variable(tf.random_normal([num_hidden_2,num_hidden_1]))
    b3 = tf.Variable(tf.random_normal([num_hidden_1]))
    w4 = tf.Variable(tf.random_normal([num_hidden_1,num_input]))
    b4 = tf.Variable(tf.random_normal([num_input]))
    
    # Layer Encoder
    layer_1 = tf.matmul(x,w1)
    layer_1_b = tf.add(layer_1 ,b1)
    layer_1_act = tf.nn.sigmoid(layer_1_b)

    layer_2 = tf.matmul(layer_1_act,w2)
    layer_2_b = tf.add(layer_2 ,b2)
    layer_2_act = tf.nn.sigmoid(layer_2_b)

    layer_3 = tf.matmul(layer_2_act,w3)
    layer_3_b = tf.add(layer_3 ,b3)
    layer_3_act = tf.nn.sigmoid(layer_3_b)

    final = tf.matmul(layer_3_act,w4)
    final_b = tf.add(final ,b4)
    final_act = tf.nn.sigmoid(final_b)

    # Loss function
    # loss_1 = tf.reduce_mean(tf.pow(final_act - x, 2))

    temp =tf.reshape(final_act,[1000,28,28,1])
    temp_2 =tf.reshape(x,[1000,28,28,1])
    value = tf_ssim(temp,temp_2)
    loss_1 = tf.cond(0 < value, lambda: value, lambda: tf.negative(value))

    # optimizer = tf.train.MomentumOptimizer(learning_rate,0.001).minimize(loss_1 )
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_1 )
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)

# 2. Make session 
sess = tf.Session(graph=graph)
with sess:

    sess.run(tf.global_variables_initializer())
    past_i = 0

    for iter in range(1, num_steps+1):
        for i in range(1000,55000,1000):
            current_x = x_data[past_i:i]
            _, l1 = sess.run([optimizer, loss_1], feed_dict={x: current_x})
            past_i = i

            if i % 1000 == 0:
                print('Current Epoch: %i, Current Batch: %i, Minibatch Mean Loss: %f,Minibatch SSIM Loss: ' % (iter,i, l1 ))
 
        past_i = 0


    # ------- TESTING CODE USED -------------
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(final_act, feed_dict={x: batch_x})
        
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















sys.exit()
# ----------------- RUNNING EXMAPLE -----------------
# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
    return layer_2

batch_size = 50

# 1. Make GRaph 
graph  = tf.Graph()
with graph.as_default():

    X = tf.placeholder("float", [None, num_input])

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([num_input])),
    }

    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)


# 2. Make session 
sess = tf.Session(graph=graph)
with sess:

    # 2.1 Declare the variable
    sess.run(tf.global_variables_initializer())

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))


    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})
        
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


# ----------------- END CODE ---------------