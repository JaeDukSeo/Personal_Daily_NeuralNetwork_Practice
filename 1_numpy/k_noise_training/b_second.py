import numpy as np,sys
from mnist import MNIST
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

np.random.seed(452)

def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * (1 - log(x))

def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1/( 1 + x **2)

# 0. Load Data
mndata = MNIST('../../z_MINST_DATA')
images, labels = mndata.load_testing()
images,labels = np.array(images),np.array(labels)
only_zero_index,only_one_index = np.where(labels==0)[0],np.where(labels==1)[0]

only_zero_image,only_zero_label = images[[only_zero_index]],np.expand_dims(labels[[only_zero_index]],axis=1)
only_one_image,only_one_label   = images[[only_one_index]],np.expand_dims(labels[[only_one_index]],axis=1)

images = np.vstack((only_zero_image,only_one_image))
labels = np.vstack((only_zero_label,only_one_label))
images,label = shuffle(images,labels)

testing_images, testing_lables =images[:20,:],label[:20,:]
training_images,training_lables =images[20:,:],label[20:,:]

learning_rate = 0.3
learning_rate2 = 0.00001
learning_rate3 = 0.0001
time = 4
time2 = 240
num_epoch = 100
value  = 0.2
w1 = np.random.randn(784,840)*value
w2 = np.random.randn(840,1024)*value
w3 = np.random.randn(1024,1)*value

b1 = np.random.randn(2095,840) *value
b2 = np.random.randn(2095,1024)*value
b3 = np.random.randn(2095,1)*value

w1n,w2n,w3n = w1,w2,w3
b1n,b2n,b3n = b1,b2,b3

w1l,w2l,w3l = w1,w2,w3
b1l,b2l,b3l = b1,b2,b3

w1l2,w2l2,w3l2 = w1,w2,w3
b1l2,b2l2,b3l2 = b1,b2,b3



for iter in range(num_epoch):
    
    l1 = training_images.dot(w1l2) + b1l2
    l1A = arctan(l1)

    l2 = l1A.dot(w2l2)+ b2l2
    l2A = arctan(l2)

    l3 = l2A.dot(w3l2)+ b3l2
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')

    w3lg = np.random.gumbel(size=w3l2.shape)
    b3lg = np.random.gumbel(size=b3l2.shape)

    w2lg = np.random.gumbel(size=w2l2.shape)
    b2lg = np.random.gumbel(size=b2l2.shape)

    w1lg = np.random.gumbel(size=w1l2.shape)
    b1lg = np.random.gumbel(size=b1l2.shape)
    
    if iter < time:
        w3l2 = w3l2 + 0.001 * learning_rate* cost * w3lg
        b3l2 = b3l2 + 0.001 *learning_rate* cost * b3lg

        w2l2 = w2l2 + 0.01 *learning_rate* cost * w2lg
        b2l2 = b2l2 + 0.01 * learning_rate*cost * b2lg

        w1l2 = w1l2 + 0.1 * learning_rate*cost * w1lg
        b1l2 = b1l2 + 0.1 * learning_rate*cost * b1lg
    
    if iter > time:
        learning_rate = learning_rate2
        w3l2 = w3l2 + 0.001 * learning_rate* cost * w3lg
        b3l2 = b3l2 + 0.001 *learning_rate* cost * b3lg

        w2l2 = w2l2 + 0.01 *learning_rate* cost * w2lg
        b2l2 = b2l2 + 0.01 * learning_rate*cost * b2lg

        w1l2 = w1l2 + 0.1 * learning_rate*cost * w1lg
        b1l2 = b1l2 + 0.1 * learning_rate*cost * b1lg
    
    if iter > time2:
        learning_rate = learning_rate3



print('\n\n')
l1 = testing_images.dot(w1l2) + b1l2
l1A = arctan(l1)

l2 = l1A.dot(w2l2)+ b2l2
l2A = arctan(l2)

l3 = l2A.dot(w3l2)+ b3l2
l3A = log(l3)

print(testing_lables.T)
print(np.round(np.squeeze(l3A).T))
print(np.squeeze(l3A).T)

print('\n\n')


learning_rate = 0.3
learning_rate2 = 0.001
learning_rate3 = 0.0001
time = 2
time2 = 24
num_epoch = 100
value  = 0.2


for iter in range(num_epoch):
    
    l1 = training_images.dot(w1l) + b1l
    l1A = arctan(l1)

    l2 = l1A.dot(w2l)+ b2l
    l2A = arctan(l2)

    l3 = l2A.dot(w3l)+ b3l
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')

    w3lg = np.random.gumbel(size=w3l.shape)
    b3lg = np.random.gumbel(size=b3l.shape)

    w2lg = np.random.gumbel(size=w2l.shape)
    b2lg = np.random.gumbel(size=b2l.shape)

    w1lg = np.random.gumbel(size=w1l.shape)
    b1lg = np.random.gumbel(size=b1l.shape)
    
    w3l = w3l + 0.001 * learning_rate* cost * w3lg
    b3l = b3l + 0.001 *learning_rate* cost * b3lg

    w2l = w2l + 0.01 *learning_rate* cost * w2lg
    b2l = b2l + 0.01 * learning_rate*cost * b2lg

    w1l = w1l + 0.1 * learning_rate*cost * w1lg
    b1l = b1l + 0.1 * learning_rate*cost * b1lg
    
    if iter > time:
        learning_rate = learning_rate2

    if iter > time2:
        learning_rate = learning_rate3


print('\n\n')
l1 = testing_images.dot(w1l) + b1l
l1A = arctan(l1)

l2 = l1A.dot(w2l)+ b2l
l2A = arctan(l2)

l3 = l2A.dot(w3l)+ b3l
l3A = log(l3)
print(testing_lables.T)
print(np.round(np.squeeze(l3A).T))
print(np.squeeze(l3A).T)

print('\n\n')

for iter in range(num_epoch):
    
    l1 = training_images.dot(w1n) 
    l1A = arctan(l1)

    l2 = l1A.dot(w2n)
    l2A = arctan(l2)

    l3 = l2A.dot(w3n)
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')

    grad_3_part_1 = l3A - training_lables
    grad_3_part_2 = d_log(l3)
    grad_3_part_3 = l2A
    grad_3_w = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)
    grad_3_b = grad_3_part_1 * grad_3_part_2

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3n.T)
    grad_2_part_2 = d_arctan(l2)
    grad_2_part_3 = l1A
    grad_2_w = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
    grad_2_b = grad_2_part_1 * grad_2_part_2



    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2n.T)
    grad_1_part_2 = d_arctan(l1)
    grad_1_part_3 = training_images
    grad_1_w = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)
    grad_1_b = grad_1_part_1 * grad_1_part_2

    w3n = w3n - learning_rate * grad_3_w
    b3n = b3n - learning_rate * grad_3_b

    w2n = w2n - learning_rate * grad_2_w
    b2n = b2n - learning_rate * grad_2_b

    w1n = w1n - learning_rate * grad_1_w
    b1n = b1n - learning_rate * grad_1_b

l1 = testing_images.dot(w1n) 
l1A = arctan(l1)

l2 = l1A.dot(w2n)
l2A = arctan(l2)

l3 = l2A.dot(w3n)
l3A = log(l3)
print('\n\n')

print(testing_lables.T)
print(np.round(np.squeeze(l3A).T))
print(np.squeeze(l3A).T)

print('\n\n')










# -- end code --