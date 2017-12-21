from sklearn.datasets import make_classification
from sklearn.utils import shuffle
import matplotlib.pyplot as plt, numpy as np,sys,sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics


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

# 178
np.random.seed(17228)

# 0. Data Preprocess and declare hyper parameter
X, Y = make_classification(n_samples=1800,n_features=2,
                        # n_classes = 3,
                        class_sep=0.45, n_redundant=0, 
                        n_informative=2,
                        n_clusters_per_class=1)

# X, Y = make_moons(n_samples=1500, random_state=40, noise=0.05)
x_data, y_data, x_label_og, y_label_og = train_test_split(X, Y, random_state=50)
x_label = np.expand_dims(x_label_og,axis=1)
y_label = np.expand_dims(y_label_og,axis=1)

# plt.scatter(X[:,0],X[:,1],c = Y)
# plt.show()
# plt.scatter(x_data[:,0],x_data[:,1],c = x_label_og)
# plt.show()
# plt.scatter(y_data[:,0],y_data[:,1],c = y_label_og)
# plt.show()

input_d,h1_d,h2_d,out_d  = 2,50,76,1
w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,out_d)
numer_of_epoch = 1500
past_i = 0

learning_rate = 0.1

for iter in range(numer_of_epoch):

    for i in range(100,700,100):
        
        current_x_batch =  x_data[past_i:i]
        current_y_batch =  x_label[past_i:i] 

        layer_1  = current_x_batch.dot(w1)
        layer_1_act  = sigmoid(layer_1)

        layer_2  = layer_1_act.dot(w2)
        layer_2_act  = sigmoid(layer_2)

        final  = layer_2_act.dot(w3)
        final_act  = sigmoid(final)
        
        # print "Current Mini : ",past_i," : ",i, "  Current cost : ", np.square(final_act - current_y_batch).sum() / len(current_x_batch)
        # print "Current Learning : ",learning_rate,'\n'

        past_i = i

        grad_3_past_1 = (final_act - current_y_batch)
        grad_3_past_2 = d_sigmoid(final)
        grad_3_past_3 = layer_2_act
        grad_3 = grad_3_past_3.T.dot(grad_3_past_1 * grad_3_past_2)
        
        grad_2_past_1 = (grad_3_past_1 * grad_3_past_2).dot(w3.T)
        grad_2_past_2 = d_sigmoid(layer_2)
        grad_2_past_3 = layer_1_act
        grad_2 = grad_2_past_3.T.dot(grad_2_past_1 * grad_2_past_2)

        grad_1_past_1 = (grad_2_past_1 * grad_2_past_1).dot(w2.T)
        grad_1_past_2 = d_sigmoid(layer_1)
        grad_1_past_3 = current_x_batch
        grad_1 = grad_1_past_3.T.dot(grad_1_past_1 * grad_1_past_2)


        w3 -= learning_rate * grad_3
        w2 -= learning_rate * grad_2
        w1 -= learning_rate * grad_1

    # 1. 
    # if iter == 800:
    #     learning_rate = 0.0001

    # 2. Time Based Decay
    # learning_rate = learning_rate *  1.0/(1.0 + 0.00001 * iter)

    learning_rate = learning_rate * np.exp(-0.00001 * iter)

    if iter % 100 == 0:
        past_i =0
        current_x_batch =  y_data
        current_y_batch =  y_label

        layer_1  = current_x_batch.dot(w1)
        layer_1_act  = sigmoid(layer_1)

        layer_2  = layer_1_act.dot(w2)
        layer_2_act  = sigmoid(layer_2)

        final  = layer_2_act.dot(w3)
        final_act  = sigmoid(final)

        print "Current Learning : ",learning_rate
        print "Current Epoch : ", iter, "  Test Set Error : ",np.square(final_act - current_y_batch).sum() / len(current_x_batch)
        print "Pridict : ",final_act[:3].tolist(), "  GT : ",current_y_batch[:3].tolist()
        print('Current Accuacry : ',metrics.accuracy_score(current_y_batch, np.round(final_act)))
        print(metrics.confusion_matrix(current_y_batch, np.round(final_act))),'\n\n'













# ---------- END CODE ------