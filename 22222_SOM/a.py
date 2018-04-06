import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

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

PathDicom = "../z_CIFAR_data/cifar10/cifar-10-batches-py/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if not ".html" in filename.lower() and not  ".meta" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

global train_batch,train_label,test_batch,test_label

# Read the data and reshape and noramlize data
batch0 = unpickle(lstFilesDCM[0])[b'data']
batch0 = (batch0 - batch0.min(axis=0))/(batch0.max(axis=0)-batch0.min(axis=0))
images = np.reshape(batch0,(10000,3,32,32))
images = np.rot90(np.rot90(images,1,axes=(1,3)),3,axes=(1,2)).astype(np.float32)




# ---- end code ---