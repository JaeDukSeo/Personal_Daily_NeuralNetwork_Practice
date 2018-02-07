import numpy as np,sys,time
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(223)

mnist = input_data.read_data_sets('../MNIST_data', one_hot=False)
data = mnist.test.images






















# ---- end code ---