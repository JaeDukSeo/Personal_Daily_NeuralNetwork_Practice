import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

def logistic(x):
    return 1 / ( 1 + np.exp( -1 * x )  )

def d_logistic(x):
    return logistic(x) * (1 - logistic(x))


def LReLu(matrix):
    mask  = (matrix<=0) * 0.01
    mask2 = (matrix>0) * 1.0
    final_mask = mask + mask2
    return final_mask * matrix

def d_LReLu(matrix):
    mask  = (matrix<=0) * 0.01
    mask2 = (matrix>0) * 1.0
    return mask + mask2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

# 0. Data preprocess and declare hyper parameter and parameter
np.random.seed(4)
mnist = input_data.read_data_sets("../4_tensorflow/MNIST_data/", one_hot=True)
x, x_label, y, y_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
x = x.reshape(-1, 28, 28, 1)  # 28x28x1 input img
y = y.reshape(-1, 28, 28, 1)  # 28x28x1 input img

number_of_epoch = 100
input_d,h1_d,h2_d,out_d = 784,625,420,10
leaing_rate = 0.3

w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,out_d)

# Func: Adjust the test set size
x = x[:2000]

for epoch in range(number_of_epoch):

    for i in range(0,len(x)):

        x_data_reshaped  = x[i].reshape(1,28*28)
        x_data_label = x_label[i]

        # 1. Make the operation
        layer_1 = x_data_reshaped.dot(w1)
        layer_1_act = LReLu(layer_1)
        m1 = np.random.binomial(1, 0.2, size=layer_1_act.shape)
        layer_1_act = layer_1_act * m1

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = LReLu(layer_2)
        m2 = np.random.binomial(1, 0.2, size=layer_2_act.shape)
        layer_2_act = layer_2_act * m2

        final  = layer_2_act.dot(w3)
        final_act = logistic(final)

        # 2. Make the back propagation and cost
        cost  = np.square(final_act - x_data_label).sum()

        grad_3_part_1 = 2.0 * (final_act - x_data_label)
        grad_3_part_2 = d_logistic(final)
        grad_3_part_3 = layer_2_act
        grad_3 = (grad_3_part_1 * grad_3_part_2).T.dot(grad_3_part_3).T

        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_LReLu(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3).T

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_LReLu(layer_1)
        grad_1_part_3 = x_data_reshaped
        grad_1 = (grad_1_part_1 * grad_1_part_2).T.dot(grad_1_part_3).T

        w1 = w1 - leaing_rate * grad_1 
        w2 = w2 - leaing_rate * grad_2 
        w3 = w3 - leaing_rate * grad_3

    test_num = int(np.random.uniform(low=0, high=len(y), size=(1,1)))

    x_data_reshaped = y[0].reshape(1,28*28)
    x_data_label = y_label[0]

    layer_1 = x_data_reshaped.dot(w1)
    layer_1_act = LReLu(layer_1)
    m1 = np.random.binomial(1, 0.2, size=layer_1_act.shape)
    layer_1_act = layer_1_act * m1

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = LReLu(layer_2)
    m2 = np.random.binomial(1, 0.2, size=layer_2_act.shape)
    layer_2_act = layer_2_act * m2

    final  = layer_2_act.dot(w3)
    final_act = logistic(final)

    print 'Number of Epoch: ',epoch
    print x_data_label
    print np.squeeze(final_act)







# -------- END OF THE CODE --------