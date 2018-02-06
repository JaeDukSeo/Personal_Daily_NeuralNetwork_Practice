import numpy as np
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(5678)

np.set_printoptions(precision=3,suppress=True)
def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()

temp = np.random.randn(1,10)
temp[0,0]= 1
print(temp)
print(softmax(temp))

gt = np.expand_dims(np.array([1,0,0,0,0,0,0,0,0,0]),axis=0)

print(gt.shape)

softLsay = softmax(temp)

print(gt - softLsay)


cross = - (gt * np.log(softLsay) + (1 - gt) * np.log(1-softLsay))

print(cross)




# 0. Declare Training Data and Labels
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

train = mnist.test
images, labels = train.images, train.labels
images,label = shuffle(images,labels)

test_image_num,training_image_num = 10,100
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:test_image_num + training_image_num,:],label[test_image_num:test_image_num + training_image_num,:]


for x in testing_lables:
    print(x)


def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    grad = softmax(X)

    print(grad.shape)

    grad[range(m),y] = grad[range(m),y] -  1
    grad = grad/m
    return grad

one = np.expand_dims(testing_images[0,:],axis=0)
one_abel =np.expand_dims( testing_lables[0,:],axis=1)


print(one.shape)
print(one_abel.shape)

one = one.dot(np.random.randn(784,10))
print(one.shape)


def SoftmaxLoss(X, y):
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = - (y * np.log(p))
    loss = np.sum(log_likelihood) / m

    dx = p.copy()
    dx[range(m), y] -= 1
    dx /= m
    return loss, dx

ssdsadsa = SoftmaxLoss(one,one_abel)

# -- end code --