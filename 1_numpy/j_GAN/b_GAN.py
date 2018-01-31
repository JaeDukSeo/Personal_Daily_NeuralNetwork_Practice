import numpy as np,sys
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from mnist import MNIST
import cupy as np
np.cuda.Device(0).use()
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

# 2. Declare Hyper Parameters

D_w1 = np.random.randn(784,246) * 0.002
D_w2 = np.random.randn(246,128)* 0.002
D_w3 = np.random.randn(128,1)* 0.002

G_w1 = np.random.randn(100,280)* 0.0002
G_w2 = np.random.randn(280,380)* 0.0002
G_w3 = np.random.randn(380,560)* 0.0002
G_w4 = np.random.randn(560,784)* 0.0002

def generator(Z):
    
    gl1 = Z.dot(G_w1)
    gl1A = arctan(gl1)

    gl2 = gl1A.dot(G_w2)
    gl2A = ReLu(gl2)
    
    gl3 = gl2A.dot(G_w3)
    gl3A = ReLu(gl3)

    gl4 = gl3A.dot(G_w4)
    gl4A = log(gl4)

    return gl4A

def discriminator(X):
    
    dl1 = X.dot(D_w1)
    dl1A = arctan(dl1)
    
    dl2 = dl1A.dot(D_w2)
    dl2A = arctan(dl2)

    dl3 = dl2A.dot(D_w3)
    dl3A = log(dl3)
    
    return dl3A

v1,v2,v3 = 0,0,0
v4,v5,v6,v7 = 0,0,0,0

learing_rate =   0.0001
learing_rate_G = 0.001
alpha = 0.01
num_epoch = 1000000
total_cost_D = 0

for iter in range(num_epoch):

    # Func: Train Dis 
    for k_step in range(2):
        
        batch_location = np.random.randint(len(images) - 10)
        current_image = images[batch_location:batch_location+10,:]
        current_gen = generator(np.random.uniform(-1., 1., size=[10, 100]))

        # Func: Forward Feed From Real Data
        dl1_r = current_image.dot(D_w1)
        dl1A_r = arctan(dl1_r)

        dl2_r = dl1A_r.dot(D_w2)
        dl2A_r = arctan(dl2_r)

        dl3_r = dl2A_r.dot(D_w3)
        current_real = log(dl3_r)

        # Func: Forward Feed From Fake Data
        dl1_f = current_gen.dot(D_w1)
        dl1A_f = arctan(dl1_f)

        dl2_f = dl1A_f.dot(D_w2)
        dl2A_f = arctan(dl2_f)

        dl3_f = dl2A_f.dot(D_w3)
        current_fake = log(dl3_f)

        # Func: Get the Cost
        cost = -1 *(1/10) * (np.log(current_real) + np.log(1-current_fake)).sum()
        # print("Current Iter :",iter, " current Cost Dis: ", cost,end='\r')
        total_cost_D = total_cost_D + cost

        # Func: Back Propagation Respect to Real Image
        grad_real_w3_part_1 = -1 *(1/10) * (1/current_real)
        grad_real_w3_part_2 = d_log(dl3_r)
        grad_real_w3_part_3 = dl2A_r
        grad_real_w3 = grad_real_w3_part_3.T.dot(grad_real_w3_part_1 * grad_real_w3_part_2)
        
        grad_real_w2_part_1 = (grad_real_w3_part_1 * grad_real_w3_part_2).dot(D_w3.T)
        grad_real_w2_part_2 = d_arctan(dl2_r)
        grad_real_w2_part_3 = dl1A_r
        grad_real_w2 = grad_real_w2_part_3.T.dot(grad_real_w2_part_1 * grad_real_w2_part_2)

        grad_real_w1_part_1 = (grad_real_w2_part_1 * grad_real_w2_part_1).dot(D_w2.T)
        grad_real_w1_part_2 = d_arctan(dl1_r)
        grad_real_w1_part_3 = current_image
        grad_real_w1 = grad_real_w1_part_3.T.dot(grad_real_w1_part_1 * grad_real_w1_part_2)

        # Func: Back Propagation Respect to Fake Image
        grad_fake_w3_part_1 = -1 *(1/10) * (1/(1-current_fake))
        grad_fake_w3_part_2 = d_log(dl3_f)
        grad_fake_w3_part_3 = dl2A_f
        grad_fake_w3 = grad_fake_w3_part_3.T.dot(grad_fake_w3_part_1 * grad_fake_w3_part_2)
        
        grad_fake_w2_part_1 = (grad_fake_w3_part_1 * grad_fake_w3_part_2).dot(D_w3.T)
        grad_fake_w2_part_2 = d_arctan(dl2_f)
        grad_fake_w2_part_3 = dl1A_f
        grad_fake_w2 = grad_fake_w2_part_3.T.dot(grad_fake_w2_part_1 * grad_fake_w2_part_2)

        grad_fake_w1_part_1 = (grad_fake_w2_part_1 * grad_fake_w2_part_2).dot(D_w2.T)
        grad_fake_w1_part_2 = d_arctan(dl1_f)
        grad_fake_w1_part_3 = current_gen
        grad_fake_w1 = grad_fake_w1_part_3.T.dot(grad_fake_w1_part_1 * grad_fake_w1_part_2)

        grad_D_w1 = grad_real_w1 + grad_fake_w1
        grad_D_w2 = grad_real_w2 + grad_fake_w2
        grad_D_w3 = grad_real_w3 + grad_fake_w3
        
        v1 = v1 *0.01* alpha + learing_rate * grad_D_w1
        v2 = v2 *0.01* alpha +  learing_rate * grad_D_w2
        v3 = v3 *0.01* alpha +  learing_rate * grad_D_w3

        D_w1 = D_w1 - v1
        D_w2 = D_w2 - v2
        D_w3 = D_w3 - v3

    # Func: Train Gen
    current_gen = np.random.uniform(-1., 1., size=[10, 100])

    gl1 = current_gen.dot(G_w1)
    gl1A = arctan(gl1)

    gl2 = gl1A.dot(G_w2)
    gl2A = ReLu(gl2)
    
    gl3 = gl2A.dot(G_w3)
    gl3A = ReLu(gl3)

    gl4 = gl3A.dot(G_w4)
    gl4A = log(gl4)

    cost_G = -1 *(1/10) * np.log(gl4A).sum()
    print("Current Iter :",iter, " current Cost Gen: ", cost_G," Total cost D : ",total_cost_D,end='\r')
    total_cost_D = 0

    grad_G_w4_part_1 = -1 *(1/10) * (1/gl4A)
    grad_G_w4_part_2 = d_log(gl4)
    grad_G_w4_part_3 = gl3A
    grad_G_w4 =    grad_G_w4_part_3.T.dot(grad_G_w4_part_1 * grad_G_w4_part_2)

    grad_G_w3_part_1 = (grad_G_w4_part_1 * grad_G_w4_part_2).dot(G_w4.T)
    grad_G_w3_part_2 = d_ReLu(gl3)
    grad_G_w3_part_3 = gl2A
    grad_G_w3 =   grad_G_w3_part_3.T.dot(grad_G_w3_part_1 * grad_G_w3_part_2)

    grad_G_w2_part_1 = (grad_G_w3_part_1 * grad_G_w3_part_2).dot(G_w3.T)
    grad_G_w2_part_2 = d_ReLu(gl2)
    grad_G_w2_part_3 = gl1A
    grad_G_w2 =   grad_G_w2_part_3.T.dot(grad_G_w2_part_1 * grad_G_w2_part_2)

    grad_G_w1_part_1 = (grad_G_w2_part_1 * grad_G_w2_part_2).dot(G_w2.T)
    grad_G_w1_part_2 = d_arctan(gl1)
    grad_G_w1_part_3 = current_gen
    grad_G_w1 =   grad_G_w1_part_3.T.dot(grad_G_w1_part_1 * grad_G_w1_part_2)

    v4 = v4 * alpha + learing_rate_G * grad_G_w1
    v5 = v5 * alpha + learing_rate_G * grad_G_w2
    v6 = v6 * alpha + learing_rate_G * grad_G_w3
    v7 = v7 * alpha + learing_rate_G * grad_G_w4

    G_w1 = G_w1 - v4
    G_w2 = G_w2 - v5
    G_w3 = G_w3 - v6
    G_w4 = G_w4 - v7

    if iter%200 == 0:
        current_gen = np.random.uniform(-1., 1., size=[16, 100])

        gl1 = current_gen.dot(G_w1)
        gl1A = arctan(gl1)

        gl2 = gl1A.dot(G_w2)
        gl2A = ReLu(gl2)

        gl3 = gl2A.dot(G_w3)
        gl3A = ReLu(gl3)

        gl4 = gl3A.dot(G_w4)
        gl4A = log(gl4)
    
        fig = plot(np.asnumpy(gl4A))
        plt.savefig('out/{}.png'.format(str(iter).zfill(3)), bbox_inches='tight')
        plt.close(fig)



# -- end code ---