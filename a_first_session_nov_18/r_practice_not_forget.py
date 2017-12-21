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


# ------- 3D Plots Backend Order ------
# TkAgg
# WX
# QTAgg
# QT4Agg

np.random.seed(1)
X, Y = make_classification(n_samples= 600,class_sep=2.0  ,n_classes=3,n_features=3, n_redundant=0, n_informative=2, n_clusters_per_class=1)
x_train,y_tain,x_label,y_label = train_test_split(X,Y)

print x_train.shape
print y_tain.shape
print np.expand_dims(x_label,axis=1).shape
print np.expand_dims(y_label,axis=1).shape

x_label= np.expand_dims(x_label,axis=1)
y_label= np.expand_dims(y_label,axis=1)

num_epoch = 500000
input_d, h1_d, h2_d, h3_d, out_d  = 3,5,10,5,1

w1 = np.random.randn(input_d,h1_d) * 3
w2 = np.random.randn(h1_d,h2_d)* 3
w3 = np.random.randn(h2_d,h3_d)* 3
w4 = np.random.randn(h3_d,out_d)* 3

learning_rate = 0.00001

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
        print "batch : ", i, " : ", i +50, ' Loss : ', loss

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

        w4 -= learning_rate*grad_4
        w3 -= learning_rate*grad_3
        w2 -= learning_rate*grad_2
        w1 -= learning_rate*grad_1


    print 'Current Epoch: ',iter

    layer_1 = y_tain.dot(w1)
    layer_1_act = sigmoid(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = sigmoid(layer_2)

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = sigmoid(layer_3)

    final_act = layer_3_act.dot(w4)
    # final_act = sigmoid(final)  

    loss = np.square(final_act - y_label).sum() / ( 2 * len(current_x_data))
    print 'Mini Batch Loss : ', loss,'\n\n\n'



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

# ------ end -----