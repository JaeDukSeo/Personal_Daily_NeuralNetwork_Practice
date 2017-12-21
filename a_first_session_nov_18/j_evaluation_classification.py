import numpy as np,sys,sklearn
import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit

def sigmoid(x):
    return 1/(1 + np.exp(-1*x))

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
# sklearn.datasets.make_classification
# sklearn.utils.shuffle()

# 0. Data preprocess and declare hyper parameter
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age','label']
pima = pd.read_csv(url, header=None, names=col_names)

# define X and y
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
# X is a matrix, hence we use [] to access the features we want in feature_cols
X = pima[feature_cols]
# y is a vector, hence we use dot to access 'label'
y = pima.label
x_data, y_data, x_label, y_label = train_test_split(X, y, random_state=0)


# 1. Weights declared
number_of_epoch = 300
input_d,h1_d,h2_d,h3_d,out_d = 4,300,500,700,1

w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,h3_d)
w4 = np.random.randn(h3_d,out_d)

# 1.5 Only shuffle the data once
shuffle_x = sklearn.utils.shuffle(x_data)
shuffle_x_label = np.expand_dims(sklearn.utils.shuffle(x_label),axis=1)

for iter in range(number_of_epoch):

    # 1.8 Shuffle Data each time
    # shuffle_x = sklearn.utils.shuffle(x_data)
    # shuffle_x_label = np.expand_dims(sklearn.utils.shuffle(x_label),axis=1)
    error_sum_1 = 0

    for i,j in ((0,96),(96,192),(192,288),(288,384),(384,480),(480,576)):

        current_x = shuffle_x[i:j]
        current_x_label = shuffle_x_label[i:j]

        layer_1 = current_x.dot(w1)
        layer_1_act = LReLu(layer_1)
        
        layer_2 = layer_1_act.dot(w2)
        layer_2_act = LReLu(layer_2)

        layer_3 = layer_2_act.dot(w3)
        layer_3_act = sigmoid(layer_3)

        final = layer_3_act.dot(w4)
        final_act = sigmoid(final)

        # print "Error : ", np.square(final_act - current_x_label).sum() / ( 2 * len(final_act))
        # error_sum_1+= np.square(final_act - current_x_label).sum() / ( 2 * len(final_act))

        grad_4_part_1 = (final_act - current_x_label)
        grad_4_part_2 = d_sigmoid(final)
        grad_4_part_3 = layer_3_act
        grad_4 = grad_4_part_3.T.dot(grad_4_part_1*grad_4_part_1)

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
        
        w4 -= 0.4*grad_4
        w3 -= 0.4*grad_3
        w2 -= 0.4*grad_2
        w1 -= 0.4*grad_1
    print '----------- FINISHED EPOCH : ',iter," ------------\n\n"
    print "Sum error: ",error_sum_1
        
print '--------- FINISH TRAINING ---------'
layer_1 = y_data.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = sigmoid(layer_3)

final = layer_3_act.dot(w4)
final_act = sigmoid(final)

print "Error 1 : ", np.square(final_act - np.expand_dims(y_label,axis=1)).sum() / ( 2 * len(final_act))
print "Final Error : ", metrics.accuracy_score(y_label, final_act)
print(metrics.confusion_matrix(y_label, final_act))
# ------- RESULTS ABOVE SHOWS ---- that the model is not learning anything

# ------------------- online given model -----------------
logreg = LogisticRegression()
logreg.fit(x_data, x_label)
y_pred_class = logreg.predict(y_data)
print(metrics.accuracy_score(y_label, y_pred_class))
print(metrics.confusion_matrix(y_label, y_pred_class))

# ------ end of the code -----