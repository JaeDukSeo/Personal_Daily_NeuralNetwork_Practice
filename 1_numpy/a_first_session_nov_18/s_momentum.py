import numpy  as np
import matplotlib
matplotlib.use('TkAgg') 
import sklearn 
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import  train_test_split
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    return 1 / (1 + np.exp(  -1 * x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

# ------- 3D Plots Backend Order ------
# TkAgg
# WX
# QTAgg
# QT4Agg

np.random.seed(1)
X, Y = make_classification(n_samples= 1000,class_sep=1.9  ,n_classes=3,n_features=3, n_redundant=0, n_informative=2, n_clusters_per_class=1)
x_train,y_tain,x_label,y_label = train_test_split(X,Y)


x_label= np.expand_dims(x_label,axis=1)
y_label= np.expand_dims(y_label,axis=1)

num_epoch = 9000
input_d, h1_d, h2_d, h3_d, out_d  = 3,4,8,5,1

w1 = np.random.randn(input_d,h1_d) * 3
w2 = np.random.randn(h1_d,h2_d)* 3
w3 = np.random.randn(h2_d,h3_d)* 3
w4 = np.random.randn(h3_d,out_d)* 3

v1 = 0
v2 = 0 
v3 = 0 
v4 = 0 
gamma = 0.08

learning_rate = 0.07

for iter in range(num_epoch):
    for i in range(0,len(x_train),50):

        current_x_data = x_train[i:i+50]
        current_x_label = x_label[i:i+50]
        
        layer_1 = current_x_data.dot(w1)
        layer_1_act = sigmoid(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = sigmoid(layer_2)

        layer_3 = layer_2_act.dot(w3)
        layer_3_act = sigmoid(layer_3)

        final_act = layer_3_act.dot(w4)
        # final_act = sigmoid(final)
        
        loss = np.square(final_act - current_x_label).sum() / ( 2 * len(current_x_data))
        # print "batch : ", i, " : ", i +50, ' Loss : ', loss

        grad_4_part_1 = (final_act - current_x_label) / len(current_x_data)
        # grad_4_part_2 = d_sigmoid(final)
        grad_4_part_3 = layer_3_act
        grad_4 =   grad_4_part_3.T.dot(grad_4_part_1)

        grad_3_part_1 = (grad_4_part_1).dot(w4.T)
        grad_3_part_2 = d_sigmoid(layer_3)
        grad_3_part_3 = layer_2_act
        grad_3 =    grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_sigmoid(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 =    grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = current_x_data
        grad_1 =    grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)  

        # Keep Track of the previous gradient was
        v1 = gamma*v1 + learning_rate*grad_1
        v2 = gamma*v2 + learning_rate*grad_2
        v3 = gamma*v3 + learning_rate*grad_3
        v4 = gamma*v4 + learning_rate*grad_4

        w4 -= v4
        w3 -= v3
        w2 -= v2
        w1 -= v1

    if iter == 7000:
        learning_rate = 0.08
    elif iter == 8000:
        learning_rate = 0.0001



    layer_1 = y_tain.dot(w1)
    layer_1_act = sigmoid(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = sigmoid(layer_2)

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = sigmoid(layer_3)

    final_act = layer_3_act.dot(w4)
    # final_act = sigmoid(final)  

    if iter % 1000 ==0:
        print 'Current Epoch: ',iter
        loss = np.square(final_act - y_label).sum() / ( 2 * len(current_x_data))
        print 'Mini Batch Loss : ', loss
        print 'Mini Batch Acc : ', 1.0 - loss,'\n\n\n'



layer_1 = y_tain.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = sigmoid(layer_3)

final_act = layer_3_act.dot(w4)
# final_act = np.round(sigmoid(final) )


fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.scatter(y_tain[:,0],y_tain[:,1],y_tain[:,2],c = np.squeeze(final_act))

bx = fig.add_subplot(212, projection='3d')
bx.scatter(y_tain[:,0],y_tain[:,1],y_tain[:,2],c = np.squeeze(y_label))
plt.show()







# ----- END CODE ----