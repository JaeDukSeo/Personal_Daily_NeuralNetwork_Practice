
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tqdm import tqdm

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

#  -- end code -- 