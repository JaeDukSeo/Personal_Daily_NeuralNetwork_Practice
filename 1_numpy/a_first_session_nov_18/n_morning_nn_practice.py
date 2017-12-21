import numpy as np,sys
import sklearn.datasets
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


np.random.seed(5123)

def sigmoid(x):
    return 1 / (1 + np.exp( -1 * x))

def d_sigmoid(x):
    return sigmoid(x) * ( 1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return  1 - np.square(np.tanh(x))

def elu(matrix):
    safe  = (matrix>0) * 1.0
    mask  = (matrix<=0) * 1.0
    return (safe*matrix) + 3.0 *(1 - np.exp(mask * matrix))

def d_elu(matrix):
    safe = (matrix>0) * 1.0
    mask2 = (matrix<=0) * 1.0
    temp = matrix * mask2
    final = (3.0 * np.exp(temp))*mask2
    return (matrix * safe) + final

def LReLu(matrix):
    safe  = (matrix>0) * 1.0
    mask  = (matrix<=0) * 0.01
    return (matrix*safe) + (matrix*mask)

def d_LReLu(matrix):
    safe  = (matrix>0) * 1.0
    mask  = (matrix<=0) * 0.01
    return safe + mask

# 0. Data Preprocess and declare HP
# x, y = sklearn.datasets.make_moons(n_samples=800, random_state=1)
x, y = make_classification(n_samples=800,n_features=2, n_classes= 2,  class_sep = 6.0, n_redundant=0, n_informative=1,n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.0625, random_state=42)

input_d, h1_d, h2_d, h3_d, h4_d, h5_d, out_d = 2,32,67,89,82,100,1

w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,h3_d)
w4 = np.random.randn(h3_d,h4_d)
w5 = np.random.randn(h4_d,h5_d)
w6 = np.random.randn(h5_d,out_d)

# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()

# plt.scatter(X_test[:,0],X_test[:,1],c=y_test)
# plt.show()

# sys.exit()
learning_rate = 0.005
numer_of_epoch = 10

shuffle_x,shuffle_y = sklearn.utils.shuffle(X_train,y_train)
shuffle_y = np.expand_dims(shuffle_y,axis=1)

# 1. Declare operations
for iter in range(numer_of_epoch):

    shuffle_x,shuffle_y = sklearn.utils.shuffle(X_train,y_train)
    shuffle_y = np.expand_dims(shuffle_y,axis=1)

    for i,j in ((0,50),(50,100),(100,150) ):

        current_x = shuffle_x[i:j]
        current_y = shuffle_y[i:j]

        layer_1 = current_x.dot(w1)
        layer_1_act = elu(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = tanh(layer_2)

        layer_3 = layer_2_act.dot(w3)
        layer_3_act = elu(layer_3)

        layer_4 = layer_3_act.dot(w4)
        layer_4_act = tanh(layer_4)

        layer_5 = layer_4_act.dot(w5)
        layer_5_act = tanh(layer_5)

        final = layer_5_act.dot(w6)
        final_act = sigmoid(final)

        if iter %200 == 0 :
            print "Current Error : ", np.square(final_act.astype(np.int64) - current_y).sum() / (2 * len(final_act))
            print "Current Error : ", np.square(final_act - current_y).sum() / (2 * len(final_act))
            print "Error Raw: ",(final_act.astype(np.int64) - current_y).sum()
            print "Error Raw2: ",(final_act - current_y).sum()
            

        grad_6_part_1 = (final_act - current_y)
        grad_6_part_2 = d_sigmoid(final)
        grad_6_part_3 = layer_5_act
        grad_6 = grad_6_part_3.T.dot(grad_6_part_1 * grad_6_part_2)

        grad_5_part_1 = (grad_6_part_1 * grad_6_part_2).dot(w6.T)
        grad_5_part_2 = d_tanh(layer_5)
        grad_5_part_3 = layer_4_act
        grad_5 = grad_5_part_3.T.dot(grad_5_part_1 * grad_5_part_2)

        grad_4_part_1 = (grad_5_part_1 * grad_5_part_2).dot(w5.T)
        grad_4_part_2 = d_tanh(layer_4)
        grad_4_part_3 = layer_3_act
        grad_4 = grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

        grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
        grad_3_part_2 = d_elu(layer_3)
        grad_3_part_3 = layer_2_act
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)
        
        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_tanh(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_elu(layer_1)
        grad_1_part_3 = current_x
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w6 -= learning_rate*grad_6
        w5 -= learning_rate*grad_5
        w4 -= learning_rate*grad_4
        w3 -= learning_rate*grad_3
        w2 -= learning_rate*grad_2
        w1 -= learning_rate*grad_1
        
X_test,y_test = sklearn.utils.shuffle(X_train,y_train)
current_x = X_test
current_y = np.expand_dims(y_test,axis=1)

layer_1 = current_x.dot(w1)
layer_1_act = elu(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = tanh(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = elu(layer_3)

layer_4 = layer_3_act.dot(w4)
layer_4_act = tanh(layer_4)

layer_5 = layer_4_act.dot(w5)
layer_5_act = tanh(layer_5)

final = layer_5_act.dot(w6)
final_act = sigmoid(final)

final_act = final_act.astype(np.int64)


# plt.figure(1)

# plt.subplot(211)
# plt.scatter(current_x[:,0],current_x[:,1],c=np.squeeze(final_act))

# plt.subplot(212)
# plt.scatter(current_x[:,0],current_x[:,1],c=np.squeeze(current_y))

# plt.show()

print(metrics.accuracy_score(current_y, final_act))
print(metrics.confusion_matrix(current_y, final_act))

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.confusion_matrix(y_test, y_pred_class))


# plt.figure(1)

# plt.subplot(211)
# plt.scatter(current_x[:,0],current_x[:,1],c=np.squeeze(y_pred_class))

# plt.subplot(212)
# plt.scatter(current_x[:,0],current_x[:,1],c=np.squeeze(current_y))

# plt.show()





learning_rate = 0.4
input_d, h1_d,out_d = 2,32,1
numer_of_epoch =1000
w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,out_d)

# 1. Declare operations
for iter in range(numer_of_epoch):

    shuffle_x,shuffle_y = sklearn.utils.shuffle(X_train,y_train)
    shuffle_y = np.expand_dims(shuffle_y,axis=1)

    for i,j in ((0,50),(50,100),(100,150) ):

        current_x = shuffle_x[i:j]
        current_y = shuffle_y[i:j]

        layer_1 = current_x.dot(w1)
        layer_1_act = elu(layer_1)

        final = layer_1_act.dot(w2)
        final_act = sigmoid(final)

        if iter %200 == 0 :
            print "Current Error : ", np.square(final_act.astype(np.int64) - current_y).sum() / (2 * len(final_act))
            print "Current Error : ", np.square(final_act - current_y).sum() / (2 * len(final_act))
            print "Error Raw: ",(final_act.astype(np.int64) - current_y).sum()
            print "Error Raw2: ",(final_act - current_y).sum()
            

        grad_2_part_1 = (final_act - current_y)
        grad_2_part_2 = d_sigmoid(final)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_elu(layer_1)
        grad_1_part_3 = current_x
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w2 -= learning_rate*grad_2
        w1 -= learning_rate*grad_1


X_test,y_test = sklearn.utils.shuffle(X_train,y_train)
current_x = X_test
current_y = np.expand_dims(y_test,axis=1)

layer_1 = current_x.dot(w1)
layer_1_act = elu(layer_1)

final = layer_1_act.dot(w2)
final_act = sigmoid(final)
final_act = final_act.astype(np.int64)

plt.figure(1)

plt.subplot(211)
plt.scatter(current_x[:,0],current_x[:,1],c=np.squeeze(final_act))

plt.subplot(212)
plt.scatter(current_x[:,0],current_x[:,1],c=np.squeeze(current_y))

plt.show()

print(metrics.accuracy_score(current_y, final_act))
print(metrics.confusion_matrix(current_y, final_act))

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.confusion_matrix(y_test, y_pred_class))


plt.figure(1)

plt.subplot(211)
plt.scatter(current_x[:,0],current_x[:,1],c=np.squeeze(y_pred_class))

plt.subplot(212)
plt.scatter(current_x[:,0],current_x[:,1],c=np.squeeze(current_y))

plt.show()










sys.exit()

number_x = np.linspace(-1.0, 2.0, num=50)
number_y = np.linspace(-0.4, 1.0, num=50)
number_data = np.vstack((number_x,number_y)).T

current_x = number_data

layer_1 = current_x.dot(w1)
layer_1_act = elu(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = tanh(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = elu(layer_3)

final = layer_3_act.dot(w4)
final_act = np.round(tanh(final))



# ---------- END OF TEH CDOE 