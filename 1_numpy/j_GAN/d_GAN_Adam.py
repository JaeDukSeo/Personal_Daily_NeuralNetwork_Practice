import numpy as np,sys
from sklearn.utils import shuffle
from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(32432)

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

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig




# 1. Load Data and declare hyper
mndata = MNIST()
images, labels = mndata.load_testing()
images, labels = shuffle(np.asarray(images),np.asarray(labels))
num_epoch = 1

# 2. Declare Weights
D_W1 = np.random.normal(size=(784,128),scale=(1. / np.sqrt(784 / 2.)))
B_b1 = np.random.normal(size=(128),scale=(1. / np.sqrt(128 / 2.)))
D_W2 = np.random.normal(size=(128,1),scale=(1. / np.sqrt(128 / 2.)))
B_b2 = np.random.normal(size=(1),scale=(1. / np.sqrt(1 / 2.)))

G_W1 = np.random.normal(size=(100,128),scale=(1. / np.sqrt(100 / 2.)))
G_b1 = np.random.normal(size=(128),scale=(1. / np.sqrt(128 / 2.)))
G_W2 = np.random.normal(size=(128,784),scale=(1. / np.sqrt(128 / 2.)))
G_b2 = np.random.normal(size=(784),scale=(1. / np.sqrt(784 / 2.)))

for iter in range(num_epoch):

    random_int = np.random.randint(len(images) - 10)
    current_image = images[random_int]

    # Func: Generate The first Fake Data
    Z = np.random.uniform(-1., 1., size=[1, 100])
    Gl1 = Z.dot(G_W1) + G_b1
    Gl1A = ReLu(Gl1)
    Gl2 = Gl1A.dot(G_W2) + G_b2
    current_fake_data = log(Gl2)

    # Func: Forward Feed for Real data
    Dl1_r = current_image.dot(D_W1) + B_b1
    Dl1_rA = ReLu(Dl1_r)
    Dl2_r = Dl1_rA.dot(D_W2) + B_b2
    Dl2_rA = log(Dl2_r)

    # Func: Forward Feed for Fake Data
    Dl1_f = current_fake_data.dot(D_W1) + B_b1
    Dl1_fA = ReLu(Dl1_f)
    Dl2_f = Dl1_fA.dot(D_W2) + B_b2
    Dl2_fA = log(Dl2_f)

    # Func: Cost D
    D_cost = -np.log(Dl2_rA) - np.log(1.0- Dl2_fA)

    # Func: Gradient
    print(D_cost)








    # Func: Forward Feed for G
    Z = np.random.uniform(-1., 1., size=[1, 100])
    Gl1 = Z.dot(G_W1) + G_b1
    Gl1A = ReLu(Gl1)
    Gl2 = Gl1A.dot(G_W2) + G_b2
    current_fake_data = log(Gl2)

    Dl1 = current_fake_data.dot(D_W1) + B_b1
    Dl1_A = ReLu(Dl1)
    Dl2 = Dl1_A.dot(D_W2) + B_b2
    Dl2_A = log(Dl2)

    # Func: Cost G
    G_cost = -np.log(Dl2_A)
    # Func: Gradient
    print(G_cost)






# -- end code --