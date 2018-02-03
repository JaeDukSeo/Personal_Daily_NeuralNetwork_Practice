import numpy as np,sys
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(678)

def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2 

def ReLu(x):
    mask = (x > 0.0) * 1.0
    return x * mask
def d_ReLu(x):
    mask = (x > 0.0) * 1.0
    return mask    


# 0. Declare Training Data and Labels
mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=False)
train = mnist.test
images, labels = train.images, train.labels
only_zero_index,only_one_index = np.where(labels==0)[0],np.where(labels==1)[0]
only_zero_image,only_zero_label = images[[only_zero_index]],np.expand_dims(labels[[only_zero_index]],axis=1)
only_one_image,only_one_label   = images[[only_one_index]],np.expand_dims(labels[[only_one_index]],axis=1)

images = np.vstack((only_zero_image,only_one_image))
labels = np.vstack((only_zero_label,only_one_label))
images,label = shuffle(images,labels)

test_image_num = 10
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num+50,:],label[test_image_num:test_image_num+50,:]

# 1. Declare the Hidden States and weights and hyper parameters
num_epoch = 50
h = np.zeros((images.shape[1],3))
wx = np.random.randn(784,784)  * 0.2
w_rec = np.random.randn(784,784)  * 0.2
w_fc = np.random.randn(784,1)* 0.2

w_sg_1 = np.random.randn(784,784)* 0.2
w_sg_2 = np.random.randn(784,784)* 0.2

lr_wx = 0.001
lr_wrec = 0.00001
lr_sg = 0.0001
total_cost = 0

for iter in range(num_epoch):
    
    for current_image_index in range(len(training_images)):
        
        current_image = training_images[current_image_index]
        current_label = training_lables[current_image_index]

        l1 = h[:,0].dot(w_rec) + current_image.dot(wx)
        l1A = tanh(l1)
        h[:,1] = l1A

        # ----- Time Stamp 1 Syn Grad Update ------
        grad_1sg_part_1 = l1A.dot(w_sg_1)
        grad_1sg_part_2 = d_tanh(l1)
        grad_1sg_part_rec = h[:,0]
        grad_1sg_part_x = current_image

        grad_1sg_rec = grad_1sg_part_rec.T.dot(grad_1sg_part_1 * grad_1sg_part_2)
        grad_1sg_x = grad_1sg_part_x.T.dot(grad_1sg_part_1 * grad_1sg_part_2)
        
        w_rec = w_rec + lr_wrec * grad_1sg_rec
        wx = wx + lr_wrec * grad_1sg_x
        grad_true_0 = (grad_1sg_part_1 * grad_1sg_part_2).dot(w_rec.T)
        # ----- Time Stamp 1 Syn Grad Update ------
        
        l2 = h[:,1].dot(w_rec) + current_image.dot(wx)
        l2A = tanh(l2)
        h[:,2] = l2A

        # ----- Time Stamp 2 Syn Grad Update ------
        grad_2sg_part_1 = l2A.dot(w_sg_2)
        grad_2sg_part_2 = d_tanh(l2)
        grad_2sg_part_rec = h[:,1]
        grad_2sg_part_x = current_image

        grad_2sg_rec = grad_2sg_part_rec.T.dot(grad_2sg_part_1 * grad_2sg_part_2)
        grad_2sg_x = grad_2sg_part_x.T.dot(grad_2sg_part_1 * grad_2sg_part_2)
        
        w_rec = w_rec + lr_wrec * grad_2sg_rec
        wx = wx + lr_wrec * grad_2sg_x
        grad_true_1_from_2 = (grad_2sg_part_1 * grad_2sg_part_2).dot(w_rec.T)
        # ----- Time Stamp 2 Syn Grad Update ------

        # ----- Time Stamp 1 True Gradient Update ------
        grad_true_1_part_1 = grad_1sg_part_1 - grad_true_1_from_2
        grad_true_1_part_2 = h[:,1]
        grad_true_1 = np.expand_dims(grad_true_1_part_2,axis=0).T.dot(np.expand_dims(grad_true_1_part_1,axis=0))
        w_sg_1 = w_sg_1 - lr_sg * grad_true_1
        # ----- Time Stamp 1 True Gradient Update ------

        # ----- Fully Connected for Classification ------
        l3 = h[:,2].dot(w_fc)
        l3A = log(l3)
        cost = np.square(l3A - current_label).sum() * 0.5
        total_cost = total_cost + cost
        # ----------------------------------------------

        # ------- FC weight update ---------------------
        grad_fc_part_1 = l3A - current_label
        grad_fc_part_2 = d_log(l3)
        grad_fc_part_3 = h[:,2]
        grad_fc = np.expand_dims(grad_fc_part_3,axis=0).T.dot(np.expand_dims((grad_fc_part_1 * grad_fc_part_2),axis=0))
        w_fc = w_fc - lr_wx * grad_fc

        grad_true_2_from_3 = (grad_fc_part_1 * grad_fc_part_2).dot(w_fc.T)
        # ------- FC weight update ---------------------
        
        # ----- Time Stamp 2 True Gradient Update ------
        grad_true_2_part_1 = grad_2sg_part_1 - grad_true_2_from_3
        grad_true_2_part_2 = h[:,2]
        grad_true_2 = np.expand_dims(grad_true_2_part_2,axis=0).T.dot(np.expand_dims(grad_true_2_part_1,axis=0))
        w_sg_2 = w_sg_2 - lr_sg * grad_true_2
        # ----- Time Stamp 2 True Gradient Update ------

    print("current iter : ",iter, " Total current cost: ",total_cost,end='\n')
    total_cost = 0















print('\n-------------------')
print("Training Done Final Results")
predict = np.array([])
for current_image_index in range(len(testing_images)):
    current_image = testing_images[current_image_index]

    l1 = h[:,0].dot(w_rec) + current_image.dot(wx)
    l1A = tanh(l1)
    h[:,1] = l1A

    l2 = h[:,1].dot(w_rec) + current_image.dot(wx)
    l2A = tanh(l2)
    h[:,2] = l2A

    l3 = h[:,2].dot(w_fc)
    l3A = log(l3)

    predict = np.append(predict,l3A)

for i in range(len(predict)): 
    print("Results : ", predict[i],"Results Rounded: ", np.round(predict[i]), " GT: ", testing_lables[i])
print('-------------------\n')
# ------------ Normal Gate RNN Train -------





# -- end code --