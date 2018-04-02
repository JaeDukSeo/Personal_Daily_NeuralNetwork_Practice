import tensorflow as tf
import numpy as np
import os,sys
import scipy.misc
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

tf.set_random_seed(789)
np.random.seed(568)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)

class conlayer():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.005))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,stride=1,dilate=1,add=True):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],dilations=[1,dilate,dilate,1],padding='SAME')
        self.layerB = tf.nn.batch_normalization(self.layer,scale=True,offset=True,mean=0.0,variance_epsilon=1e-8,variance=1.0)
        self.layer_add =  self.layerB + self.layer
        if add:
            self.layerA = tf_relu(self.layer_add) + self.input
        else:
            self.layerA = tf_relu(self.layer_add)
        return self.layerA

data_location = "./c_preprocessed_data/train/"
train_files = []  # create an empty list
for dirName, subdirList, fileList in os.walk(data_location):
    train_files = fileList

data_location = "./c_preprocessed_data/mask/"
train_label = []  # create an empty list
for dirName, subdirList, fileList in os.walk(data_location):
    train_label = fileList

data_location = "./c_preprocessed_data/test/"
test_files = []  # create an empty list
for dirName, subdirList, fileList in os.walk(data_location):
    test_files = fileList

train_images = np.zeros((670,256,256,1))
train_labels = np.zeros((670,256,256,1))
test_images  = np.zeros((65,256,256,1))

print(train_images.sum())
print(train_labels.sum())
print(test_images.sum())

for files in range(len(train_files)):
    train_images[files,:,:,:] = np.expand_dims(scipy.misc.imread("./c_preprocessed_data/train/"+train_files[files],'F'),axis=3)

for files in range(len(train_label)):
    train_labels[files,:,:,:] = np.expand_dims(scipy.misc.imread("./c_preprocessed_data/mask/"+train_label[files],'F'),axis=3)

for files in range(len(test_files)):
    test_images[files,:,:,:] = np.expand_dims(scipy.misc.imread("./c_preprocessed_data/test/"+test_files[files],'F'),axis=3)

print(train_images.sum())
print(train_labels.sum())
print(test_images.sum())

train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min())
test_images = (test_images - test_images.min()) / (test_images.max() - test_images.min())

# --- hyper ---
num_epoch = 500
init_lr = 0.00000001
batch_size = 10

# --- make class ---    
l1 = conlayer(3,1,3)
l2 = conlayer(3,3,3)
l3 = conlayer(3,3,3)
l4 = conlayer(3,3,3)
l5 = conlayer(3,3,3)
l6 = conlayer(3,3,3)
l7 = conlayer(3,3,3)
l8 = conlayer(3,3,3)
l9 = conlayer(3,3,3)
l10 = conlayer(3,3,3)

l11 = conlayer(1,31,1)
l12 = conlayer(1,31,1)
l13 = conlayer(1,31,1)

# --- make graph ---
x = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)

layer1B = tf.nn.batch_normalization(x,scale=True,offset=True,mean=0.0,variance=1.0,variance_epsilon=1e-8)
layer1_Input = layer1B + x

layer1 = l1.feedforward(layer1_Input)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2,dilate=1)
layer4 = l4.feedforward(layer3,dilate=1)
layer5 = l5.feedforward(layer4,stride=1,dilate=1)
layer6 = l6.feedforward(layer5,dilate=1)
layer7 = l7.feedforward(layer6,dilate=1)
layer8 = l8.feedforward(layer7,stride=1,dilate=1)
layer9 = l9.feedforward(layer8,dilate=1)
layer10 = l10.feedforward(layer9,dilate=1)

layer_concat = tf.concat([x,layer1,layer2,layer3,layer4,layer5,
                          layer6,layer7,layer8,layer9,layer10],axis=3)
# layer_drop = tf.nn.dropout(layer_concat,0.8)

layer11 = l11.feedforward(layer_concat)
layer12 = l12.feedforward(layer11)
layer13 = l13.feedforward(layer12,add=False)

final_soft = tf.sigmoid(layer13)
cost = tf.reduce_mean(tf.square(final_soft-y))
# auto_train = tf.train.AdamOptimizer(learning_rate=init_lr).minimize(cost)
auto_train = tf.train.MomentumOptimizer(learning_rate=init_lr,momentum=0.6).minimize(cost)

# --- start session ---
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        # train
        for current_batch_index in range(0,len(train_images),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_label})
            print(' Iter: ', iter, " Cost:  %.32f"% sess_results[0],end='\r')
        print('\n-----------------------')
        train_images,train_labels = shuffle(train_images,train_labels)


        test_example = np.expand_dims(train_images[4,:,:,:],axis=0)
        test_example_gt = train_labels[4,:,:,:]
        sess_results = sess.run(final_soft,feed_dict={x:test_example})
        sess_results = (sess_results - sess_results.min())/(sess_results.max() - sess_results.min())

        plt.imshow(np.squeeze(test_example),cmap='gray')
        plt.pause(0.5)

        plt.imshow(np.squeeze(test_example_gt),cmap='gray')
        plt.pause(0.5)

        plt.clf()        
        plt.title("Relst")
        plt.imshow(np.squeeze(sess_results),cmap='gray')
        plt.pause(1)

        sess_results = (sess_results - sess_results.min())/(sess_results.max() - sess_results.min())
        plt.clf()
        plt.title("Relst normalized")
        plt.imshow(np.squeeze(sess_results),cmap='gray')
        plt.pause(0.5)



# -- end code --