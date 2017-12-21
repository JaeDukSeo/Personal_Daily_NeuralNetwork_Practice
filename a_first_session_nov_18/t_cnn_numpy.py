import numpy as np
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn
from sklearn.datasets import load_digits
import matplotlib
matplotlib.use('TkAgg') 
import sklearn 
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import  train_test_split
from matplotlib import pyplot as plt
from scipy import signal
np.random.seed(1)
from scipy import fftpack

def convolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft*psf_fft)))

def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))



def sigmoid(x):
    return 1  / (1 + np.exp(-1 * x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

digits =load_digits()
images= digits.images
label =  digits.target

image_train = images[:797]
image_test = images[797:]

image_train_label = label[:797]
image_test_label = label[797:]

w1 = np.random.randn(3,3)
w2 = np.random.randn(3,3)

w3 = np.random.randn(4*4,5)
w4 = np.random.randn(5,1)

for iter in range(0,len(image_train[:1])):

    current_x_data = image_train[iter]
    current_label = np.expand_dims(image_train_label[iter],axis=1)

    layer_1 = signal.convolve2d(current_x_data, w1,  mode='valid')
    layer_1_act = sigmoid(layer_1)

    layer_2 = signal.convolve2d(layer_1_act, w2, mode='valid')
    layer_2_act = sigmoid(layer_2)

    layer_3_data = layer_2_act.reshape((1,4*4))
    layer_3 = layer_3_data.dot(w3)
    layer_3_act = sigmoid(layer_3)

    final = layer_3_act.dot(w4)

    grad_4_part_1 = (final  - current_label) / 1
    grad_4_part_3 = layer_3_act
    grad_4 = grad_4_part_3.T.dot(grad_4_part_1) 

    grad_3_part_1 = grad_4_part_1.dot(w4.T)
    grad_3_part_2 =  d_sigmoid(layer_3)
    grad_3_part_3 =   layer_2_act.reshape((1,4*4))
    grad_3   = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)     

    grad_2_part_1 = (grad_3_part_1* grad_3_part_2).dot(w3.T).reshape((4,4))
    grad_2_part_2 =    d_sigmoid(layer_2)
    grad_2_part_3 =     layer_1_act
    grad_2 = signal.convolve2d(grad_2_part_3,(grad_2_part_1 * grad_2_part_2), mode='valid')

    grad_1_part_1,s = signal.deconvolve( (grad_2_part_1 * grad_2_part_2)[-1,:],w2.T[-1,:])
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = current_x_data
    # grad_1 = 

    print (grad_2_part_1 * grad_2_part_2).shape
    print w2.shape
    print    grad_1_part_2.shape  
    






# ----- END CODE -----