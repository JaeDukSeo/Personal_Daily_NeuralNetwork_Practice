import numpy as np,sys
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from mnist import MNIST
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
num_epoch = 500
learning_rate = 0.000001 

D_w1 = np.random.randn(784,128) * 0.0002
D_w2 = np.random.randn(128,1) *  0.0002

G_w1 = np.random.randn(100,128) *  0.0002
G_w2 = np.random.randn(128,784) *  0.0002


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

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

for iter in range(num_epoch):

    for batch_size in range(0,len(images_train),10):
        
        current_image = images_train[batch_size:batch_size+10,:]
        curren_generated = sample_Z(10,100)
        
        G_l1 = curren_generated.dot(G_w1)
        G_h1 = ReLu(G_l1)
        G_log_prob = G_h1.dot(G_w2)
        G_prob = log(G_log_prob)

        D_l1_r = current_image.dot(D_w1)
        D_h1_r = ReLu(D_l1_r)
        D_logit_r = D_h1_r.dot(D_w2)
        D_real = log(D_logit_r)

        D_l1_f = G_prob.dot(D_w1)
        D_h1_f = ReLu(D_l1_f)
        D_logit_f = D_h1_f.dot(D_w2)
        D_fake = log(D_logit_f)

        D_loss = -1 * np.log(D_real) - np.log( 1 - D_fake)
        G_loss = -1 * np.log(D_fake)

        print("Current iter: ",iter, "  Current batch: ",batch_size, " D loss: ",D_loss.sum(), " G Loss: ", G_loss.sum(),end='\r')


        grad_D_part_r_1 = -1 * (1/D_real) 
        grad_D_part_r_2 = d_log(D_logit_r)
        grad_D_part_r_3 = D_h1_r
        grad_D_r = grad_D_part_r_3.T.dot(grad_D_part_r_1 * grad_D_part_r_2)

        grad_D_part_f_1 = -1 * (1/( 1 - D_fake)) 
        grad_D_part_f_2 = d_log(D_logit_f)
        grad_D_part_f_3 = D_h1_f
        grad_D_f = grad_D_part_f_3.T.dot(grad_D_part_f_1 * grad_D_part_f_2)
        grad_D_w2 = grad_D_r + grad_D_f



        grad_D_part_r_w1_1 = (grad_D_part_r_1 * grad_D_part_r_2).dot(D_w2.T)
        grad_D_part_r_w1_2 = d_ReLu(D_l1_r)
        grad_D_part_r_w1_3 = current_image
        grad_D_r_w1 = grad_D_part_r_w1_3.T.dot(grad_D_part_r_w1_1 * grad_D_part_r_w1_2)

        grad_D_part_f_w1_1 = (grad_D_part_f_1 * grad_D_part_f_2).dot(D_w2.T)
        grad_D_part_f_w1_2 = d_ReLu(D_l1_f)
        grad_D_part_f_w1_3 = G_prob
        grad_D_f_w1 = grad_D_part_f_w1_3.T.dot(grad_D_part_f_w1_1 * grad_D_part_f_w1_2)
        grad_D_w1 = grad_D_r_w1 + grad_D_f_w1


        grad_G_part_1 = ((-1 * (1/D_fake) * (1/D_logit_f)).dot(D_w2.T) * d_ReLu(D_l1_f)).dot(D_w1.T)
        grad_G_part_2 = d_log(G_log_prob)
        grad_G_part_3 = G_h1
        grad_G_w2 = grad_G_part_3.T.dot(grad_G_part_1 * grad_G_part_2)

        grad_G_part_1_w1 = (grad_G_part_1 * grad_G_part_2).dot(G_w2.T)
        grad_G_part_2_w1 = d_ReLu(G_l1)
        grad_G_part_3_w1 = curren_generated
        grad_G_w1 = grad_G_part_3_w1.T.dot(grad_G_part_1_w1 * grad_G_part_2_w1)


        D_w2 = D_w2 - learning_rate * grad_D_w2
        D_w1 = D_w1 - learning_rate * grad_D_w1

        G_w1 = G_w1 - learning_rate * grad_G_w1
        G_w2 = G_w2 - learning_rate * grad_G_w2

    if iter % 10 == 0:
        
        curren_generated = sample_Z(16,100)
        
        G_l1 = curren_generated.dot(G_w1)
        G_h1 = ReLu(G_l1)
        G_log_prob = G_h1.dot(G_w2)
        G_prob = log(G_log_prob)

        fig = plot(G_prob)
        plt.savefig('out/{}.png'.format(str(iter).zfill(3)), bbox_inches='tight')
        plt.close(fig)
# -- end code --
