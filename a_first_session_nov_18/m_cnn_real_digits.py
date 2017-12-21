import numpy as np
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn
from scipy import signal


def sigmoid(x):
    return 1 / (1 + np.exp( -1 * x))

def d_sigmoid(x):
    return sigmoid(x) * ( 1 - sigmoid(x))

def LReLu(matrix):
    safe = (matrix>0) * 1.0
    mask = (matrix<=0) * 0.01
    return (matrix * safe) + (matrix * mask)
    
def d_LReLu(matrix):
    safe = (matrix>0) * 1.0
    mask = (matrix<=0) * 0.01
    return safe + mask
    

np.random.seed(1)

# 0. Data Declare and HP
data_set = load_digits()
label = np.expand_dims(data_set.target,axis=1)

input_d,h1_1,h1_2,h2_1,h2_2 ,h3_d,out_d  = 3,3,3,3,5,4,1

w1_1 = np.random.randn(3,3)
w1_2 = np.random.randn(3,3)

w2_1 = np.random.randn(3,3)
w2_2 = np.random.randn(3,3)

w3 = np.random.randn(32,150)
w4 = np.random.randn(150,1)

images = data_set.images

for iter in range(100):
    for image in range(0,len(images)):

        current_img = images[image]
        current_label = label[image]

        layer_1_1 = signal.convolve2d(current_img, w1_1,  mode='valid')
        layer_1_1_act  = LReLu(layer_1_1)

        layer_1_2 = signal.convolve2d(current_img, w1_2,  mode='valid')
        layer_1_2_act  = LReLu(layer_1_2)

        layer_2_1 = signal.convolve2d(layer_1_1_act, w2_1, mode='valid').reshape(16,1)
        layer_2_1_act = LReLu(layer_2_1)

        layer_2_2 = signal.convolve2d(layer_1_2_act, w2_2, mode='valid').reshape(16,1)
        layer_2_2_act = LReLu(layer_2_2)

        input_layer_3 = np.vstack((layer_2_1,layer_2_2)).T
        layer_3 = input_layer_3.dot(w3)
        layer_3_act = sigmoid(layer_3)

        final = layer_3_act.dot(w4)

        # 
        if image % 500 == 0:
            print "Epoch : ", iter,
            print "  Error : ",final-current_label
        # 

        grad_4_part_1 = final-current_label
        grad_4_part_3 = layer_3_act
        grad_4 = grad_4_part_3.T.dot(grad_4_part_1)

        grad_3_part_1 = (grad_4_part_1).dot(w4.T)
        grad_3_part_2 = d_sigmoid(layer_3)
        grad_3_part_3 = input_layer_3
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)
        
        w3 -= 0.00007*grad_3
        w4 -= 0.00007*grad_4
        
    
for image in range(0,40):

    current_img = images[image]
    current_label = label[image]

    layer_1_1 = signal.convolve2d(current_img, w1_1,  mode='valid')
    layer_1_1_act  = LReLu(layer_1_1)

    layer_1_2 = signal.convolve2d(current_img, w1_2,  mode='valid')
    layer_1_2_act  = LReLu(layer_1_2)

    layer_2_1 = signal.convolve2d(layer_1_1_act, w2_1, mode='valid').reshape(16,1)
    layer_2_1_act = LReLu(layer_2_1)

    layer_2_2 = signal.convolve2d(layer_1_2_act, w2_2, mode='valid').reshape(16,1)
    layer_2_2_act = LReLu(layer_2_2)

    input_layer_3 = np.vstack((layer_2_1,layer_2_2)).T
    layer_3 = input_layer_3.dot(w3)
    layer_3_act = sigmoid(layer_3)

    final = layer_3_act.dot(w4)


    print "Itter: ",image ," GT: ",current_label,
    print " Predict: ",final




# ------- END OF THE CODE ------