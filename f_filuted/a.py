import numpy as np,dicom,sys,os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data

np.random.randn(6789)

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
# 1. Read Data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True).test
images,label = shuffle(mnist.images,mnist.labels)


print(images.shape)
print(label.shape)
full_image = np.reshape(images[0,:],(28,28))
temp = full_image[:14,:14]

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(full_image[:14,:14],cmap='gray')
# axarr[0, 0].set_title('Axis [0,0]')
axarr[0, 0].get_xaxis().set_visible(False)
axarr[0, 0].get_yaxis().set_visible(False)


axarr[0, 1].imshow(full_image[:14,14:],cmap='gray')
# axarr[0, 1].set_title('Axis [0,1]')
axarr[0, 1].get_xaxis().set_visible(False)
axarr[0, 1].get_yaxis().set_visible(False)

axarr[1, 0].imshow(full_image[14:,:14],cmap='gray')
# axarr[1, 0].set_title('Axis [1,0]')
axarr[1, 0].get_xaxis().set_visible(False)
axarr[1, 0].get_yaxis().set_visible(False)

axarr[1, 1].imshow(full_image[14:,14:],cmap='gray')
# axarr[1, 1].set_title('Axis [1,1]')
axarr[1, 1].get_xaxis().set_visible(False)
axarr[1, 1].get_yaxis().set_visible(False)

plt.show()

print(label[0,:])


# -- end code --