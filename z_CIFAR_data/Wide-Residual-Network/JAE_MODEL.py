
import h5py
import os
import numpy as np

import os,sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)
np.random.seed(789)
tf.set_random_seed(789)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.CropAndPad(percent=(-0.25, 0.25))
], random_order=True) # apply augmenters in random order


def unpickle(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

file = h5py.File('./mean_std/mean_std_cifar_10.h5','r+') 

#Retrieves all the preprocessed training and validation\testing data from a file

X_train = file['X_train'][...]
Y_train = file['Y_train'][...]
X_val = file['X_val'][...]
Y_val = file['Y_val'][...]
X_test = file['X_test'][...]
Y_test = file['Y_test'][...]

# Unpickles and retrieves class names and other meta informations of the database
classes = unpickle('../cifar10/cifar-10-batches-py/batches.meta') #keyword for label = label_names

print("Training sample shapes (input and output): "+str(X_train.shape)+" "+str(Y_train.shape))
print("Validation sample shapes (input and output): "+str(X_val.shape)+" "+str(Y_val.shape))
print("Testing sample shapes (input and output): "+str(X_test.shape)+" "+str(Y_test.shape))


# === Augment Data ===
train_images = X_train
train_labels = Y_train
train_images_augmented = seq.augment_images(train_images)
train_images = np.concatenate((train_images,train_images_augmented),axis=0)
train_labels = np.concatenate((train_labels,train_labels),axis=0)
train_images,train_labels = shuffle(train_images,train_labels)

test_images = X_test
test_labels = Y_test


import tensorflow as tf

#Hyper Parameters!

learning_rate = 0.01
init_lr = learning_rate
batch_size = 64
epochs = 300
layers = 16
beta = 0.0001 #l2 regularization scale
#ensemble = 1 #no. of models to be ensembled (minimum: 1)

shuffle_size = 2
print_size = 1
divide_size = 4

K = 8 #(deepening factor)

n_classes = 10 # another useless step that I made due to certain reasons. 

# tf Graph input

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None,n_classes])
phase = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x,shape,strides,scope):
    # Conv2D wrapper
    with tf.variable_scope(scope+"regularize",reuse=False):
        W = tf.Variable(tf.truncated_normal(shape=shape,stddev=5e-2))
    b = tf.Variable(tf.truncated_normal(shape=[shape[3]],stddev=5e-2))
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x

def activate(x,phase):
    #wrapper for performing batch normalization and elu activation
    x = tf.contrib.layers.batch_norm(x, center=True, scale=True,variables_collections=["batch_norm_non_trainable_variables_collection"],updates_collections=None, decay=0.9,is_training=phase,zero_debias_moving_mean=True, fused=True)
    return tf.nn.elu(x)


def wideres33block(X,N,K,iw,bw,s,dropout,phase,scope):
    
    # Creates N no. of 3,3 type residual blocks with dropout that consitute the conv2/3/4 blocks
    # with widening factor K and X as input. s is stride and bw is base width (no. of filters before multiplying with k)
    # iw is input width.
    # (see https://arxiv.org/abs/1605.07146 paper for details on the block)
    # In this case, dropout = probability to keep the neuron enabled.
    # phase = true when training, false otherwise.
    
    conv33_1 = conv2d(X,[3,3,iw,bw*K],s,scope)
    conv33_1 = activate(conv33_1,phase)
    conv33_1 = tf.nn.dropout(conv33_1,dropout)
    
    conv33_2 = conv2d(conv33_1,[3,3,bw*K,bw*K],1,scope)
    conv_skip= conv2d(X,[1,1,iw,bw*K],s,scope) #shortcut connection

    
    caddtable = tf.add(conv33_2,conv_skip)
    
    #1st of the N blocks for conv2/3/4 block ends here. The rest of N-1 blocks will be implemented next with a loop.

    for i in range(0,int(N-1)):
        
        C = caddtable
        Cactivated = activate(C,phase)
        
        conv33_1 = conv2d(Cactivated,[3,3,bw*K,bw*K],1,scope)
        conv33_1 = activate(conv33_1,phase)
        
        conv33_1 = tf.nn.dropout(conv33_1,dropout)
            
        conv33_2 = conv2d(conv33_1,[3,3,bw*K,bw*K],1,scope)
        caddtable = tf.add(conv33_2,C)
    
    return activate(caddtable,phase)


    
def WRN(x,dropout,phase,layers,K,scope): #Wide residual network

    # 1 conv + 3 convblocks*(3 conv layers *1 group for each block + 2 conv layers*(N-1) groups for each block [total 1+N-1 = N groups]) = layers
    # 3*2*(N-1) = layers - 1 - 3*3
    # N = (layers -10)/6 + 1
    # So N = (layers-4)/6

    N = (layers-4)/6
    
    conv1 = conv2d(x,[3,3,3,16],1,scope)
    conv1 = activate(conv1,phase)

    conv2 = wideres33block(conv1,N,K,16,16,1,dropout,phase,scope)
    conv3 = wideres33block(conv2,N,K,16*K,32,2,dropout,phase,scope)
    conv4 = wideres33block(conv3,N,K,32*K,64,2,dropout,phase,scope)

    pooled = tf.nn.avg_pool(conv4,ksize=[1,8,8,1],strides=[1,1,1,1],padding='VALID')
    
    #Initialize weights and biases for fully connected layers
    with tf.variable_scope(scope+"regularize",reuse=False):
        wd1 = tf.Variable(tf.truncated_normal([1*1*64*K,64*K],stddev=5e-2))
        wout = tf.Variable(tf.truncated_normal([64*K, n_classes]))
    bd1 = tf.Variable(tf.constant(0.1,shape=[64*K]))
    bout = tf.Variable(tf.constant(0.1,shape=[n_classes]))

    # Fully connected layer
    # Reshape pooling layer output to fit fully connected layer input
    fc1 = tf.reshape(pooled, [-1, wd1.get_shape().as_list()[0]])   
    fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
    fc1 = tf.nn.elu(fc1)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, wout), bout)
    
    return out

# Construct model

model = WRN(x,keep_prob,phase,layers=layers,K=K,scope='1')

#l2 regularization
weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='1regularize')

regularizer=0
for i in range(len(weights)):
    regularizer += tf.nn.l2_loss(weights[i])
    

#cross entropy loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=y) + beta*regularizer)


global_step = tf.Variable(0, trainable=False)

#optimizer 
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
                                       momentum = 0.9, 
                                       use_nesterov=True).minimize(cost,global_step=global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(model,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
prediction = tf.nn.softmax(logits=model)





# === Start the Session ===
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.Session() as sess: 

  sess.run(tf.global_variables_initializer())

  train_total_cost,train_total_acc =0,0
  train_cost_overtime,train_acc_overtime = [],[]

  test_total_cost,test_total_acc = 0,0
  test_cost_overtime,test_acc_overtime = [],[]
  for iter in range(epochs):

        # Train Set
        for current_batch_index in range(0,int(len(train_images)/divide_size),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_labels[current_batch_index:current_batch_index+batch_size,:]
            sess_results =  sess.run([cost,accuracy,optimizer],
            feed_dict={x: current_batch, y: current_batch_label, keep_prob: 0.7, phase: True})
            print("current iter:", iter,' Current batach : ',current_batch_index," current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
          current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
          current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
          sess_results = sess.run([cost,accuracy],
          feed_dict={x: current_batch, y: current_batch_label, keep_prob: 0.7, phase: True})
          print("\t\t\tTest Image Current iter:", iter,' Current batach : ',current_batch_index, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
          test_total_cost = test_total_cost + sess_results[0]
          test_total_acc = test_total_acc + sess_results[1]

        # store
        train_cost_overtime.append(train_total_cost/(len(train_images)/divide_size/batch_size ) ) 
        train_acc_overtime.append(train_total_acc /(len(train_images)/divide_size/batch_size )  )
        test_cost_overtime.append(test_total_cost/(len(test_images)/batch_size ) )
        test_acc_overtime.append(test_total_acc/(len(test_images)/batch_size ) )
        
        # print
        if iter%print_size == 0:
            print('\n=========')
            print("Avg Train Cost: ", train_cost_overtime[-1])
            print("Avg Train Acc: ", train_acc_overtime[-1])
            print("Avg Test Cost: ", test_cost_overtime[-1])
            print("Avg Test Acc: ", test_acc_overtime[-1])
            print('-----------')      

        if iter%shuffle_size ==  0: 
          print("\n==== shuffling iter: =====",iter," \n")
          train_images,train_labels = shuffle(train_images,train_labels)
          test_images,test_labels = shuffle(test_images,test_labels)
          
            
        train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0

  # plot and save
  plt.figure()
  plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r',label="Train")
  plt.plot(range(len(train_cost_overtime)),test_cost_overtime,color='b',label='Test')
  plt.legend()
  plt.title('Cost over time')
  plt.show()

  plt.figure()
  plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='r',label="Train")
  plt.plot(range(len(train_acc_overtime)),test_acc_overtime,color='b',label='Test')
  plt.legend()
  plt.title('Acc over time')
  plt.show()
        