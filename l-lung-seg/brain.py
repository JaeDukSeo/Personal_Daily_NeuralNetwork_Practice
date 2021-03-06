import tensorflow as tf
import numpy as np,sys,os
from sklearn.utils import shuffle
from scipy.ndimage import imread
import matplotlib.pyplot as plt

np.random.seed(678)
tf.set_random_seed(5678)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)

class conlayer():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.005))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,stride=1,dilate=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],dilations=[1,dilate,dilate,1],padding='SAME')
        self.layerB = tf.nn.batch_normalization(self.layer,scale=True,offset=True,mean=0.0,variance_epsilon=1e-8,variance=1.0)
        self.layer_add =  self.layerB + self.layer
        self.layerA = tf_relu(self.layer_add)
        return self.layerA

# --- get data ---
data_location = "./DRIVE/2017_preprocessed/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if "original.png" in filename.lower():  # check whether the file's DICOM
            train_data.append(os.path.join(dirName,filename))

data_location = "./DRIVE/2017_preprocessed/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if "gt.png" in filename.lower():  # check whether the file's DICOM
            train_data_gt.append(os.path.join(dirName,filename))

train_images = np.zeros(shape=(460,238,238,1))
train_labels = np.zeros(shape=(460,238,238,1))

for file_index in range(len(train_data)-7):
    train_images[file_index,:,:]   = np.expand_dims(imread(train_data[file_index],mode='F',flatten=True),axis=2)
    train_labels[file_index,:,:]   = np.expand_dims(imread(train_data_gt[file_index],mode='F',flatten=True),axis=2)

train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min())

# --- hyper ---
num_epoch = 100
init_lr = 0.000001
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

l11 = conlayer(1,10,8)
l12 = conlayer(1,8,4)
l13 = conlayer(1,4,1)

# --- make graph ---
x = tf.placeholder(shape=[None,238,238,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,238,238,1],dtype=tf.float32)

layer1B = tf.nn.batch_normalization(x,scale=True,offset=True,mean=0.0,variance=1.0,variance_epsilon=1e-8)
layer1_Input = layer1B + x

layer1 = l1.feedforward(layer1_Input)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2,dilate=2)
layer4 = l4.feedforward(layer3,dilate=3)
layer5 = l5.feedforward(layer4,dilate=5)
layer6 = l6.feedforward(layer5,dilate=8)
layer7 = l7.feedforward(layer6,dilate=13)
layer8 = l8.feedforward(layer7,dilate=21)
layer9 = l9.feedforward(layer8,dilate=34)
layer10 = l10.feedforward(layer9,dilate=55)

layer_concat = tf.concat([x,layer1,layer2,layer3,layer4,layer5,
                          layer6,layer7,layer8,layer9,layer10],axis=3)
layer_drop = tf.nn.dropout(layer_concat,0.5)

layer11 = l11.feedforward(layer_drop)
layer12 = l12.feedforward(layer11)
layer13 = l13.feedforward(layer12)

final_soft = tf.sigmoid(layer13)
cost = tf.reduce_mean(tf.square(final_soft-y))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer13,labels=y))
auto_train = tf.train.AdamOptimizer(learning_rate=init_lr).minimize(cost)


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
    plt.imshow(np.squeeze(sess_results),cmap='gray')
    plt.show()

    sess_results = (sess_results - sess_results.min())/(sess_results.max() - sess_results.min())

    plt.imshow(np.squeeze(test_example),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(test_example_gt),cmap='gray')
    plt.show()

    plt.imshow(np.squeeze(sess_results),cmap='gray')
    plt.show()


# -- end code --