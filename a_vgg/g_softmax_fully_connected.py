import numpy as np,sys
import tensorflow as tf
from sklearn.utils import shuffle
from scipy import signal
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(5678)

np.set_printoptions(precision=3,suppress=True)

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2
def ReLu(x):
    mask = (x>0) * 1.0
    return mask *x
def d_ReLu(x):
    mask = (x>0) * 1.0
    return mask 
def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))
def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1 + x ** 2)

def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()

# 0. Declare Training Data and Labels
mnist_index = input_data.read_data_sets("../MNIST_data/", one_hot=False).test.labels
mnist= input_data.read_data_sets("../MNIST_data/", one_hot=True)
only_to_four = np.where(mnist_index==0)[0],np.where(mnist_index==1)[0],np.where(mnist_index==2)[0],np.where(mnist_index==3)[0],np.where(mnist_index==4)[0]


train = mnist.test
images, labels = train.images, train.labels

only_zero_image,only_zero_label = images[[only_to_four[0]]],    labels[[only_to_four[0]]][:,:5]
only_one_image,only_one_label   = images[[only_to_four[1]]],    labels[[only_to_four[1]]][:,:5]
only_two_image,only_two_label   = images[[only_to_four[2]]],    labels[[only_to_four[2]]][:,:5]
only_three_image,only_three_label = images[[only_to_four[3]]],  labels[[only_to_four[3]]][:,:5]
only_four_image,only_four_label   = images[[only_to_four[4]]],  labels[[only_to_four[4]]][:,:5]

images = np.vstack((only_zero_image,only_one_image,only_two_image,only_three_image,only_four_image))
labels = np.vstack((only_zero_label,only_one_label,only_two_label,only_three_label,only_four_label))
# images = np.vstack((only_zero_image,only_one_image))
# labels = np.vstack((only_zero_label,only_one_label))
images,label = shuffle(images,labels)


test_image_num,training_image_num = 20,600
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]





# -- end code --