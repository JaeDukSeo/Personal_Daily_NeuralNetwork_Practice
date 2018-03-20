import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

np.random.seed(6789)
tf.set_random_seed(678)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Get the Train data
data = unpickle('cifar100python/train')
train_image = data[b'data']
train_label = data[b'fine_labels']

# Get the Test Data
data = unpickle('cifar100python/test')
test_image = data[b'data']
test_label = data[b'fine_labels']

print(type(train_image))
print(dir(train_image))
print(train_image.shape)
print(train_image.min())
print(train_image.max())

print(test_image.shape)
print(test_image.min())
print(test_image.max())








# -- end code --