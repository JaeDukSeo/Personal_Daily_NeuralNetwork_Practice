import numpy as np
import tensorflow as tf
import sklearn 
import matplotlib.pyplot as plt
import sys

np.random.seed(6789)

# create random data
data = np.random.weibull(a=2,size=[10,200])
data = (2 * data + 80) * np.random.randint(99)

alpa,beta = 1.0,1.0
batch_e = 0.00001
binwidth = 1

plt.hist(data[2,:], bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()

mean = np.sum(data,axis=0)/len(data)
var  = np.sum(np.square(data-mean),axis=0)/len(data)
normalized = (data-mean)/( np.sqrt(var) + batch_e)


plt.hist(normalized[2,:], bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()