from sklearn.datasets import make_classification
from sklearn.utils import shuffle
import matplotlib.pyplot as plt, numpy as np,sys

def sigmoid(x):
    return 1/(1+ np.exp(-1*x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def LReLu(matrix):
    mask  = (matrix<=0) * 0.01
    mask2 = (matrix>0) * 1.0
    final_mask = mask + mask2
    return final_mask * matrix

def d_LReLu(matrix):
    mask  = (matrix<=0) * 0.01
    mask2 = (matrix>0) * 1.0
    return mask + mask2

np.random.seed(1)



# 0. Data Preprocess and declare hyper parameter
plt.title("One informative feature, one cluster per class", fontsize='small')
X, Y = make_classification(n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1)

input_d,h1_d,h2_d,h3_d,out_d = 2,3,3,10,1
lr = 0.4
w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,h3_d)
w4 = np.random.randn(h3_d,out_d)



for iter in range(6000):

    shuffle_x,shuffle_y = shuffle(X,Y)
    error_sum = 0

    for i,j in ((0,10),(10,20),(20,30),(30,40),(50,60),(70,80),(80,90),(90,100)):

        current_x = shuffle_x[i:j]
        current_y = np.expand_dims(shuffle_y[i:j],axis=1)

        layer_1 = current_x.dot(w1)
        layer_1_act = LReLu(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = LReLu(layer_2) 

        layer_3 = layer_2_act.dot(w3)
        layer_3_act = sigmoid(layer_3) 

        final = layer_3_act.dot(w4)
        final_act = sigmoid(final) 

        if iter%500 == 0 :
            print 'Current iter :  ',iter," Error : ",np.square(final_act-current_y).sum() / (2* len(current_x))
            error_sum +=error_sum + np.square(final_act-current_y).sum() / (2* len(current_x))

        grad_4_part_1 = (final_act-current_y)
        grad_4_part_2 = d_sigmoid(final)
        grad_4_part_3 = layer_3_act
        grad_4 = grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

        grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
        grad_3_part_2 = d_sigmoid(layer_3)
        grad_3_part_3 = layer_2_act
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_LReLu(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_LReLu(layer_1)
        grad_1_part_3 = current_x
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w4 -= lr*grad_4
        w3 -= lr*grad_3
        w2 -= lr*grad_2
        w1 -= lr*grad_1
        


layer_1 = current_x.dot(w1)
layer_1_act = LReLu(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = LReLu(layer_2) 

layer_3 = layer_2_act.dot(w3)
layer_3_act = sigmoid(layer_3) 

final = layer_3_act.dot(w4)
final_act = sigmoid(final) 
print "Final Error 1: ", (final_act-Y).sum() / (2* len(X))
print "Final Error 2: ",np.square(final_act-Y).sum() / (2* len(X))















# ------- DISPLAY DATA ----------
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y,s=25, edgecolor='k')
# plt.show()











# ------------- END OF THE CODE -------