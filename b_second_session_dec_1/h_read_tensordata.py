import os,numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets("../../4_tensorflow/MNIST_data/", one_hot=True)
X, X_label, Y, Y_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
X = X.reshape(-1, 28, 28, 1)  # 28x28x1 input img
Y = Y.reshape(-1, 28, 28, 1)  # 28x28x1 input img

print(X[1].shape)
print(X_label[3])


plt.imshow(np.squeeze(X[3]),cmap='gray')
plt.show()


# -- end code --