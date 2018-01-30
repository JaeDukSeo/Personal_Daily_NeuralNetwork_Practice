import numpy as np,sys
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

np.random.seed(45678)

import numpy , gzip
from sklearn.utils import shuffle
from mnist import MNIST
import os
cwd = os.getcwd()

print(cwd)
mndata = MNIST('')

images, labels = mndata.load_testing()

print(images.shape)


sys.exit()
# 1. Declare Data Hyper 
data = load_digits()
images = data.images
label  = data.target
label = np.expand_dims(label,axis=1)
num_epoch = 100

w1 = np.random.randn(64,128) * 0.2
w2 = np.random.randn(96,128)* 0.2
w3 = np.random.randn(128,256)* 0.2
w4 = np.random.randn(256,10)* 0.2

# 2. Encode Data
onehot_label = OneHotEncoder().fit(label)
onehot_label = onehot_label.transform(label).toarray()

print(onehot_label.shape)
print(label.shape)

print(label[0])
print(onehot_label[0,:])

print(label[1])
print(onehot_label[1,:])

print(label[100])
print(onehot_label[100,:])






# ---- end code ---