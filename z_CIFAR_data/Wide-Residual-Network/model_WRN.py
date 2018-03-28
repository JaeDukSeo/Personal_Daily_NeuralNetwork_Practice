
import h5py
import os
import numpy as np

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


classes_num = len(classes['label_names']) #classes_num = no. of classes

# Here, I am creating a special variable X_train_F which is basically a nested list.
# The outermost list of X_train_F will be a list of all the class values (0-9 where each value correspond to a class name)
# Each elements (class values) of the outermost list is actually also a list; a list of all the example data belonging
# to the particular class which corresponds to class value under which the data is listed. 

X_train_F = []

for i in range(0,classes_num):
    X_train_F.append([])


for i in range(0,len(X_train)):
    l = np.argmax(Y_train[i]) #l for label (in this case it's basically the index of class value elemenmts)  
    #(Y_train is one hot encoded. Argmax returns the index for maximum value which should be 1 and
    # that index should indicate the value)
    X_train_F[l].append(X_train[i])

import random


smoothing_factor = 0.1 #for label smoothing

def create_batches(batch_size,classes_num):
   
    s = int(batch_size/classes_num) #s denotes samples taken from each class to create the batch.
    no_of_batches = int(len(X_train)/batch_size)
    
    shuffled_indices_per_class =[]
    for i in range(0,classes_num):
        temp = np.arange(len(X_train_F[i]))
        np.random.shuffle(temp)
        shuffled_indices_per_class.append(temp)
        
    batches_X = []
    batches_Y = []
        
    for i in range(no_of_batches):
        
        shuffled_class_indices = np.arange(classes_num)
        np.random.shuffle(shuffled_class_indices)
        
        batch_Y = np.zeros((batch_size,classes_num),np.float32)
        batch_X = np.zeros((batch_size,32,32,3),np.float32)
        
        for index in range(0,classes_num):
            class_index = shuffled_class_indices[index]
            for j in range(0,s):
                batch_X[(index*s)+j] = X_train_F[class_index][shuffled_indices_per_class[class_index][i*s+j]] # Assign the s chosen random samples to the training batch
                batch_Y[(index*s)+j][class_index] = 1
                batch_Y[(index*s)+j] = (1-smoothing_factor)*batch_Y[(index*s)+j] + smoothing_factor/classes_num
        
        rs = batch_size - s*classes_num #rs denotes no. of random samples from random classes to take
                                        #in order to fill the batch if batch isn't divisble by classes_num
        #fill the rest of the batch with random data
        rand = random.sample(np.arange(len(X_train)),rs)
        j=0
        for k in range(s*classes_num,batch_size):
            batch_X[k] = X_train[int(rand[j])]
            batch_Y[k] = Y_train[int(rand[j])]
            batch_Y[k] = (1-smoothing_factor)*batch_Y[k] + smoothing_factor/classes_num
            j+=1

        batches_X.append(batch_X)
        batches_Y.append(batch_Y)
    
    return batches_X,batches_Y

batches_X,batches_Y = create_batches(64,classes_num) # A demo of the function at work

# Since each batch will have almost equal no. of cases from each class, no batch should be biased towards some particular classes
sample = random.randint(0,len(batches_X))

def random_crop(img):
    #result = np.zeros_like((img))
    c = np.random.randint(0,5)
    if c==0:
        crop = img[4:32,0:-4]
    elif c==1:
        crop = img[0:-4,0:-4]
    elif c==2:
        crop = img[2:-2,2:-2]
    elif c==3:
        crop = img[4:32,4:32]
    elif c==4:
        crop = img[0:-4,4:32]
    
    #translating cropped position
    #over the original image
    c = np.random.randint(0,5)
    if c==0:
        img[4:32,0:-4] = crop[:]
    elif c==1:
        img[0:-4,0:-4] = crop[:]
    elif c==2:
        img[2:-2,2:-2] = crop[:]
    elif c==3:
        img[4:32,4:32] = crop[:]
    elif c==4:
        img[0:-4,4:32] = crop[:]
        
    return img

def augment_batch(batch_X): #will be used to modify images realtime during training (real time data augmentation)
    
    aug_batch_X = np.zeros((len(batch_X),32,32,3))
   
    for i in range(0,len(batch_X)):
        
        hf = np.random.randint(0,2)
        
        if hf == 1: #hf denotes horizontal flip. 50-50 random chance to apply horizontal flip on images,
            batch_X[i] = np.fliplr(batch_X[i])
       
        # Remove the below cropping to apply random crops. But before that it's better to implement something like mirror padding
        # or any form of padding to increase the dimensions beforehand.
        
        c = np.random.randint(0,3)
        if c==1:
           #one in a three chance for cropping
           #randomly crop 28x28 portions and translate it.
            aug_batch_X[i] = random_crop(batch_X[i])
        else:
            aug_batch_X[i] = batch_X[i]
    
    return aug_batch_X
    
aug_batches_X=[]
for batch in batches_X:
    aug_batch_X = augment_batch(batch)
    aug_batches_X.append(aug_batch_X)

def shuffle_batch(batch_X,batch_Y):
    shuffle = random.sample(np.arange(0,len(batch_X),1,'int'),len(batch_X))
    shuffled_batch_X = []
    shuffled_batch_Y = []
    
    for i in range(0,len(batch_X)):
        shuffled_batch_X.append(batch_X[int(shuffle[i])])
        shuffled_batch_Y.append(batch_Y[int(shuffle[i])])
    
    shuffled_batch_X = np.array(shuffled_batch_X)
    shuffled_batch_Y = np.array(shuffled_batch_Y)

    return shuffled_batch_X,shuffled_batch_Y

s_batches_X=[]
s_batches_Y=[]
for i in range(len(aug_batches_X)):
    s_batch_X,s_batch_Y = shuffle_batch(aug_batches_X[i],batches_Y[i])
    s_batches_X.append(s_batch_X)
    s_batches_Y.append(s_batch_Y)

def batch(batch_size): #one shortcut function to execute all necessary functions to create a training batch
    batches_X,batches_Y = create_batches(batch_size,classes_num)
    
    aug_batches_X=[]
    for batch in batches_X:
        aug_batch_X = augment_batch(batch)
        aug_batches_X.append(aug_batch_X)
        
    s_batches_X=[]
    s_batches_Y=[]
    
    for i in range(len(aug_batches_X)):
        s_batch_X,s_batch_Y = shuffle_batch(aug_batches_X[i],batches_Y[i])
        s_batches_X.append(s_batch_X)
        s_batches_Y.append(s_batch_Y)
    
    return s_batches_X,s_batches_Y

import tensorflow as tf

#Hyper Parameters!

learning_rate = 0.01
init_lr = learning_rate
batch_size = 64
epochs = 500
layers = 16
beta = 0.0001 #l2 regularization scale
#ensemble = 1 #no. of models to be ensembled (minimum: 1)

K = 8 #(deepening factor)

n_classes = classes_num # another useless step that I made due to certain reasons. 

# tf Graph input

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None,classes_num])
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

    for i in range(0,N-1):
        
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

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() # Prepares variable for saving the model
    sess.run(init) #initialize all variables
    step = 1   
    loss_list=[]
    acc_list=[]
    val_loss_list=[]
    val_acc_list=[]
    best_val_acc=0
    total_loss=0
    total_acc=0
    avg_loss=0
    avg_acc=0
    val_batch_size = batch_size
    
    
    threshold = 0.5 #if training accuracy is 100-threshold or less, training will stop 
    
    while step <= epochs:
        
        
        # A little bit of Learning rate scheduling
        if step == 60:
            learning_rate = 0.01
        elif step == 120:
            learning_rate = 0.004
        elif step == 160:
            learning_rate = 0.0008

        
        batches_X, batches_Y = batch(batch_size)
        
        for i in range(len(batches_X)):
            # Run optimization operation (backpropagation)
            _,loss,acc = sess.run([optimizer,cost,accuracy],
                                   feed_dict={x: batches_X[i], y: batches_Y[i], 
                                   keep_prob: 0.7,
                                   phase: True})
            total_loss += loss
            total_acc += acc
            
            if i%100 == 0:
                print("Iter " + str((step-1)*len(batches_X)+i+1) + ", Minibatch Loss= " + \
                  "{:.3f}".format(loss) + ", Minibatch Accuracy= " + \
                  "{:.3f}%".format(acc*100))

     
                      
        total_val_loss=0
        total_val_acc=0
        val_loss=0
        val_acc=0
        avg_val_loss=0
        avg_val_acc=0
            
        i=0
        count=0
        while i<len(X_val):

            if i+val_batch_size<len(X_val):
                val_loss, val_acc = sess.run([cost, accuracy], 
                                            feed_dict={x: X_val[i:i+val_batch_size],
                                                       y: Y_val[i:i+val_batch_size],
                                                       keep_prob: 1,
                                                       phase: False})
            else:
                val_loss, val_acc = sess.run([cost, accuracy], 
                                            feed_dict={x: X_val[i:],
                                                       y: Y_val[i:],
                                                       keep_prob: 1,
                                                       phase: False})
                              
            total_val_loss = total_val_loss + val_loss
            total_val_acc = total_val_acc + val_acc
            count+=1
                
            i+=val_batch_size
  

 
        avg_val_loss = total_val_loss/count # Average validation loss
        avg_val_acc = total_val_acc/count # Average validation accuracy
            
             
        val_loss_list.append(avg_val_loss) # Storing values in list for plotting later on.
        val_acc_list.append(avg_val_acc) # Storing values in list for plotting later on.
            
        avg_loss = total_loss/len(batches_X) # Average mini-batch training loss
        avg_acc = total_acc/len(batches_X)   # Average mini-batch training accuracy
        loss_list.append(avg_loss) # Storing values in list for plotting later on.
        acc_list.append(avg_acc) # Storing values in list for plotting later on.
            
        total_loss=0
        total_acc=0

        print("\nEpoch " + str(step) + ", Validation Loss= " + \
                "{:.3f}".format(avg_val_loss) + ", validation Accuracy= " + \
                "{:.3f}%".format(avg_val_acc*100)+"")
        print("Epoch " + str(step) + ", Average Training Loss= " + \
                "{:.3f}".format(avg_loss) + ", Average Training Accuracy= " + \
                "{:.3f}%".format(avg_acc*100)+"")
                    
        if avg_val_acc >= best_val_acc: # When better accuracy is received than previous best validation accuracy
                
            best_val_acc = avg_val_acc # update value of best validation accuracy received yet.
            saver.save(sess, 'Model_Backup/model.ckpt') # save_model including model variables (weights, biases etc.)
            print("Checkpoint created!")
                
                
            
        if (100-(avg_acc*100)) <= threshold:
            print("\nConvergence Threshold Reached!")
            break
              
        step += 1
        
    print("\nOptimization Finished!\n")
    
    print("Best Validation Accuracy: %.3f%%"%((best_val_acc)*100))
    
    print('Loading pre-trained weights for the model...')
    saver = tf.train.Saver()
    saver.restore(sess, 'Model_Backup/model.ckpt')
    sess.run(tf.global_variables())
    print('\nRESTORATION COMPLETE\n')
    
    print('Testing Model Performance...')
    test_batch_size = batch_size
    total_test_loss=0
    total_test_acc=0
    test_loss=0
    test_acc=0
    avg_test_loss=0
    avg_test_acc=0
            
    i=0
    count=0
    while i<len(X_test):
        
        if (i+test_batch_size)<len(X_test):
            test_loss, test_acc = sess.run([cost, accuracy], 
                                         feed_dict={x: X_test[i:i+test_batch_size],
                                                    y: Y_test[i:i+test_batch_size],
                                                    keep_prob: 1,
                                                    phase: False})
        else:
            test_loss, test_acc = sess.run([cost, accuracy], 
                                            feed_dict={x: X_test[i:],
                                                       y: Y_test[i:],
                                                       keep_prob: 1,
                                                       phase: False})
   
        total_test_loss = total_test_loss+test_loss
        total_test_acc = total_test_acc+test_acc
        count+=1
        
        i+=test_batch_size
             
    avg_test_loss = total_test_loss/count # Average test loss
    avg_test_acc = total_test_acc/count # Average test accuracy
    
    print("Test Loss = " + \
          "{:.3f}".format(avg_test_loss) + ", Test Accuracy = " + \
          "{:.3f}%".format(avg_test_acc*100))