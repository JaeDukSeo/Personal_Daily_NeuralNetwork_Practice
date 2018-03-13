import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from mnist import MNIST
import matplotlib.pyplot as plt

# Make Class
class Residual_Dilated_CNN():
    
    def __init__(self,width,height,inc,outc):
        
        self.w = tf.Variable(tf.random_normal([width,height,inc,outc]))
        self.w_dilated = tf.Variable(tf.random_normal([width,height,inc,outc]))

    def getw(self): return [self.w,self.w_dilated]

    def feedforward(self,input=None,OG=None):
        return 8

    def backprop(self,gradient = None,OG=None):
        return 8

# Process data
mndata = MNIST('./')
mndata.gz = True
images, labels = mndata.load_training()
images = np.asarray(images)
labels = np.asarray(labels)
print(images.shape)
print(labels.shape)

images, labels = mndata.load_testing()
images = np.asarray(images)
labels = np.asarray(labels)
print(images.shape)
print(labels.shape)

# Create Class Object


# Create Graph


# Create session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())







# -- end code --