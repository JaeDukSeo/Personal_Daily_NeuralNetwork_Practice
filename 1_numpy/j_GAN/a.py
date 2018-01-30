import numpy as np,sys
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from mnist import MNIST
import matplotlib.pyplot as plt

np.random.seed(45678)

def ReLu(x):
    mask = (x>0) * 1.0
    return mask *x
def d_ReLu(x):
    mask = (x>0) * 1.0
    return mask 

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

# 2. Hyper parameters
num_epoch = 1
learning_rate = 0.1 

D_w1 = np.random.randn(784,128) * 0.2
D_w2 = np.random.randn(128,1) * 0.2

G_w1 = np.random.randn(100,128)
G_w2 = np.random.randn(128,784)

def generator(z):
    G_l1 = z.dot(G_w1)
    G_h1 = ReLu(G_l1)
    G_log_prob = G_h1.dot(G_w2)
    G_prob = log(G_log_prob)
    return G_prob

def discriminator(x):
    D_l1 = x.dot(D_w1)
    D_h1 = ReLu(D_l1)
    D_logit = D_h1.dot(D_w2)
    D_prob = log(D_logit)
    return D_prob

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


for iter in range(num_epoch):

    for batch_size in range(0,len(images_test),10):
        
        current_image = images_test[batch_size:batch_size+10,:]
        curren_generated = sample_Z(10,100)
        
        G_sample = generator(curren_generated)

        D_real = discriminator(current_image)
        D_fake = discriminator(G_sample)

        D_loss = -1 * np.mean(np.log(D_real) + np.log( 1 - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))



        sys.exit()


# -- end code --
