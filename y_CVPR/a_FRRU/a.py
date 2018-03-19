import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

print(mnist.train.images.shape)


# --- end code ---