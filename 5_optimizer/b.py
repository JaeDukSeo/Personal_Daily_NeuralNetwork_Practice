import numpy as np,sys
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from mnist import MNIST
import matplotlib.pyplot as plt
np.random.seed(45678)

def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1 + x ** 2)

def log(x):
    return 1 / ( 1+ np.exp(-1*x))
def d_log(x):
    return log(x) * (1 - log(x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 1. Load Data and declare hyper
mndata = MNIST()
images, labels = mndata.load_testing()
images, labels = shuffle(np.asarray(images),np.asarray(labels))
labels = np.expand_dims(labels,axis=1)
# 1.5 One hot encode
onehot_label = OneHotEncoder().fit(labels)
onehot_label = onehot_label.transform(labels).toarray()

images_test,labels_test  = images[:1000,:],onehot_label[:1000]
images_train,labels_train  = images[1000:,:],onehot_label[1000:]

w1 = np.random.randn(784,32) * 0.2
w2 = np.random.randn(32,64)* 0.2
w3 = np.random.randn(64,10)* 0.2
num_epoch = 100
learning_rate = 0.1 

for iter in range(num_epoch):
    
    for batch_size in range(0,len(images_train),10):

        current_image = images_train[batch_size:batch_size+10,:]
        current_label = labels_train[batch_size:batch_size+10,:]

        l1 = current_image.dot(w1)
        l1A = log(l1)

        l2 = l1A.dot(w2)
        l2A = log(l2)

        l3 = l2A.dot(w3)
        l3A = log(l3)

        out = softmax(l3A)
        cost = np.square(out - current_label).sum() * 0.5
        print("current iter: ",iter," Current Batch", batch_size ," current cost: ",cost,end='\r')

        grad_3_part_1 = out - current_label
        grad_3_part_2 = d_log(l3)
        grad_3_part_3 = l2A
        grad_3 =  grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

        grad_2_part_1 = (grad_3_part_1* grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_log(l2)
        grad_2_part_3 = l1A
        grad_2 =   grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1* grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_log(l1)
        grad_1_part_3 = current_image
        grad_1 =     grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)    

        w1 = w1 - learning_rate * grad_1
        w2 = w2 - learning_rate * grad_2 
        w3 = w3 - learning_rate * grad_3 

print('\n\n')
l1 = images_test.dot(w1)
l1A = arctan(l1)

l2 = l1A.dot(w2)
l2A = arctan(l2)

l3 = l2A.dot(w3)
l3A = arctan(l3)

out = softmax(l3A)
cost = np.square(out - labels_test).sum() * 0.5
print("current iter: ",iter, " current cost: ",cost,end='\r')



# -- end code --