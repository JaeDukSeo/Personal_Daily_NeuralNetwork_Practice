import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(68)
tf.set_random_seed(5678)

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_softmax(x): return tf.nn.softmax(x)

# Function to unpcicle
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

PathDicom = "../z_CIFAR_data/cifar10batchespy/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if not ".html" in filename.lower() and not  ".meta" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# Read the data traind and Test
batch0 = unpickle(lstFilesDCM[0])
batch1 = unpickle(lstFilesDCM[1])
batch2 = unpickle(lstFilesDCM[2])
batch3 = unpickle(lstFilesDCM[3])
batch4 = unpickle(lstFilesDCM[4])
train_batch = np.vstack((batch0[b'data'],batch1[b'data'],batch2[b'data'],batch3[b'data'],batch4[b'data']))
train_label = np.expand_dims(np.hstack((batch0[b'labels'],batch1[b'labels'],batch2[b'labels'],batch3[b'labels'],batch4[b'labels'])).T,axis=1).astype(np.float32)
test_batch = unpickle(lstFilesDCM[5])[b'data']
test_label = np.expand_dims(np.array(unpickle(lstFilesDCM[5])[b'labels']),axis=0).T.astype(np.float32)

# Normalize data from 0 to 1
train_batch = (train_batch - train_batch.min(axis=0))/(train_batch.max(axis=0)-train_batch.min(axis=0))
test_batch = (test_batch - test_batch.min(axis=0))/(test_batch.max(axis=0)-test_batch.min(axis=0))

# reshape data
train_batch = np.reshape(train_batch,(len(train_batch),3,32,32))
test_batch = np.reshape(test_batch,(len(test_batch),3,32,32))

# rotate data
train_batch = np.rot90(np.rot90(train_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)
test_batch = np.rot90(np.rot90(test_batch,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)

# cnn
class CNNLayer():
    
    def __init__(self,kernel,inchan,outchan):
        self.w = tf.Variable(tf.random_normal([kernel,kernel,inchan,outchan]))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))


    def feedforward(self,input,stride_num=1):
        self.input = input
        self.layer = tf.nn.conv2d(input,self.w,strides=[1,stride_num,stride_num,1],padding="VALID")
        return self.layer

# fcc
class FCCLayer():
    
    def __init__(self,input,output):
        self.w = tf.Variable(tf.random_normal([input,output]))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))
        
# create layers
l1 = CNNLayer(5,3,24)
l2 = CNNLayer(5,24,36)
l3 = CNNLayer(5,36,48)

l4 = CNNLayer(3,48,64)
l5 = CNNLayer(3,64,64)



testsss = train_batch[0:10,:,:,:].astype(np.float32)

# Create graph
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)

# create Session
with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    layer1 = l1.feedforward(testsss,stride_num=2).eval()
    layer1 = d_tf_Relu(layer1)
    print(layer1.shape)

    layer2 = l2.feedforward(layer1).eval()
    print(layer2.shape)

    layer3 = l3.feedforward(layer2).eval()
    print(layer3.shape)

    layer4 = l4.feedforward(layer3).eval()
    print(layer4.shape)

    layer5 = l5.feedforward(layer4).eval()
    print(layer5.shape)


# --- end code --