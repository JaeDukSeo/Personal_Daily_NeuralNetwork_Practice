import numpy as np
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn

def sigmoid(x):
    return 1 / (1 + np.exp( -1*x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def LReLu(mattrix):
    safe = (mattrix>0) * 1.0
    mask = (mattrix<=0) * 0.01
    return (mattrix * safe) + (mattrix* mask)
    
def d_LReLu(mattrix):
    safe = (mattrix>0) * 1.0
    mask = (mattrix<=0) * 0.01
    return safe + mask 

np.random.seed(1)

# 0. Load the Sample data set - Choose between digit and sample
# dataset = load_sample_images() 
dataset = load_digits() 
lables = np.expand_dims(dataset.target,axis=1)
current_imgs =dataset.images[:1000]
current_lables =lables[:1000]


# print current_lables[:10]
number_of_epoch = 4000
input_d,h1_d,h2_d,out_d  = 64,334, 482,1

w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,out_d)

# print current_imgs.shape
# print current_imgs[0].reshape(1,64).shape
# print len(current_imgs)

for iter in range(number_of_epoch):

    current_imgs,current_lables = sklearn.utils.shuffle(current_imgs,current_lables)

    for i,j in ( (0,200),(200,400),(400,600),(600,800),(800,1000) ):

        current_x = current_imgs[i:j].reshape(200,64)
        current_y = current_lables[i:j]

        layer_1 = current_x.dot(w1)
        layer_1_act = sigmoid(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = sigmoid(layer_2)

        final = layer_2_act.dot(w3)
        final_act = LReLu(final)

        grad_3_part_1 = (final- np.round(current_y))
        # grad_3_part_2 = d_LReLu(final)
        grad_3_part_2 = 1
        grad_3_part_3 = layer_2_act
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1*grad_3_part_2)

        grad_2_part_1 = (grad_3_part_1*grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_sigmoid(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = current_x
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w1 -= 0.00005* grad_1
        w2 -= 0.00005* grad_2
        w3 -= 0.00005* grad_3
        

        

    layer_1 = current_imgs.reshape(1000,64).dot(w1)
    layer_1_act = sigmoid(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = sigmoid(layer_2)

    final = layer_2_act.dot(w3)
    # final_act = sigmoid(final)

    print "Epoch : ",iter," Batch from:  ",i," TO: ",j
    print "Error : ",np.square(current_lables - final).sum() / ( len(current_lables))
    print "------------"



layer_1 = current_imgs[:15].reshape(15,64).dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

final = layer_2_act.dot(w3)
# final_act = sigmoid(final)



print "predict : ",np.round(final),
print "GT : ",np.round(current_lables[:15])

# print "Error : ",np.square(current_lables[:15] - final)
print "Error sum: ",np.square(np.round(current_lables[:15]) - final).sum()

# print('epoch : ',iter,'    ',metrics.accuracy_score(current_lables, final))













# -------------- END OF THT EOCDE ----- 