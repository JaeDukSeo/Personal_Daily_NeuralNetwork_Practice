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
num_epoch = 900000000
learing_rate = 0.000001
hidden_input,G_input = 800,100

# 2. Declare Weights
D_W1 = np.random.normal(size=(784,hidden_input),scale=(1. / np.sqrt(784 / 2.)))   *0.002
# D_b1 = np.random.normal(size=(128),scale=(1. / np.sqrt(128 / 2.)))       *0.002
D_b1 = np.zeros(hidden_input)

D_W2 = np.random.normal(size=(hidden_input,1),scale=(1. / np.sqrt(hidden_input / 2.)))     *0.002
# D_b2 = np.random.normal(size=(1),scale=(1. / np.sqrt(1 / 2.)))           *0.002
D_b2 = np.zeros(1)

G_W1 = np.random.normal(size=(G_input,hidden_input),scale=(1. / np.sqrt(G_input / 2.)))   *0.002
# G_b1 = np.random.normal(size=(128),scale=(1. / np.sqrt(128 / 2.)))      *0.002
G_b1 = np.zeros(hidden_input)

G_W2 = np.random.normal(size=(hidden_input,784),scale=(1. / np.sqrt(hidden_input / 2.)))  *0.002
# G_b2 = np.random.normal(size=(784),scale=(1. / np.sqrt(784 / 2.)))      *0.002
G_b2 = np.zeros(784)



# 3. For Adam Optimzier
v1,m1 = 0,0
v2,m2 = 0,0
v3,m3 = 0,0
v4,m4 = 0,0

v5,m5 = 0,0
v6,m6 = 0,0
v7,m7 = 0,0
v8,m8 = 0,0

beta_1,beta_2,eps = 0.9,0.999,0.00000001

for iter in range(num_epoch):

    random_int = np.random.randint(len(images) - 10)
    current_image = np.expand_dims(images[random_int],axis=0)

    # Func: Generate The first Fake Data
    Z = np.random.uniform(-1., 1., size=[1, G_input])
    Gl1 = Z.dot(G_W1) + G_b1
    Gl1A = arctan(Gl1)
    Gl2 = Gl1A.dot(G_W2) + G_b2
    current_fake_data = log(Gl2)

    # Func: Forward Feed for Real data
    Dl1_r = current_image.dot(D_W1) + D_b1
    Dl1_rA = ReLu(Dl1_r)
    Dl2_r = Dl1_rA.dot(D_W2) + D_b2
    Dl2_rA = log(Dl2_r)

    # Func: Forward Feed for Fake Data
    Dl1_f = current_fake_data.dot(D_W1) + D_b1
    Dl1_fA = ReLu(Dl1_f)
    Dl2_f = Dl1_fA.dot(D_W2) + D_b2
    Dl2_fA = log(Dl2_f)

    # Func: Cost D
    D_cost = -np.log(Dl2_rA) - np.log(1.0- Dl2_fA)

    # Func: Gradient
    grad_f_w2_part_1 =  - 1/(1.0- Dl2_fA)
    grad_f_w2_part_2 =  d_log(Dl2_f)
    grad_f_w2_part_3 =   Dl1_fA
    grad_f_w2 =       grad_f_w2_part_3.T.dot(grad_f_w2_part_1 * grad_f_w2_part_2) 
    grad_f_b2 = grad_f_w2_part_1 * grad_f_w2_part_2

    grad_f_w1_part_1 =  (grad_f_w2_part_1 * grad_f_w2_part_2).dot(D_W2.T)
    grad_f_w1_part_2 =  d_ReLu(Dl1_f)
    grad_f_w1_part_3 =   current_fake_data
    grad_f_w1 =       grad_f_w1_part_3.T.dot(grad_f_w1_part_1 * grad_f_w1_part_2) 
    grad_f_b1 =      grad_f_w1_part_1 * grad_f_w1_part_2

    grad_r_w2_part_1 =  - 1/Dl2_rA
    grad_r_w2_part_2 =  d_log(Dl2_r)
    grad_r_w2_part_3 =   Dl1_rA
    grad_r_w2 =       grad_r_w2_part_3.T.dot(grad_r_w2_part_1 * grad_r_w2_part_2) 
    grad_r_b2 =       grad_r_w2_part_1 * grad_r_w2_part_2

    grad_r_w1_part_1 =  (grad_r_w2_part_1 * grad_r_w2_part_2).dot(D_W2.T)
    grad_r_w1_part_2 =  d_ReLu(Dl1_r)
    grad_r_w1_part_3 =   current_image
    grad_r_w1 =       grad_r_w1_part_3.T.dot(grad_r_w1_part_1 * grad_r_w1_part_2) 
    grad_r_b1 =       grad_r_w1_part_1 * grad_r_w1_part_2

    grad_w1 =grad_f_w1 + grad_r_w1
    grad_b1 =grad_f_b1 + grad_r_b1
    
    grad_w2 =grad_f_w2 + grad_r_w2
    grad_b2 =grad_f_b2 + grad_r_b2

    # ---- Update Gradient ----
    m1 = beta_1 * m1 + (1 - beta_1) * grad_w1
    v1 = beta_2 * v1 + (1 - beta_2) * grad_w1 ** 2

    m2 = beta_1 * m2 + (1 - beta_1) * grad_b1
    v2 = beta_2 * v2 + (1 - beta_2) * grad_b1 ** 2

    m3 = beta_1 * m3 + (1 - beta_1) * grad_w2
    v3 = beta_2 * v3 + (1 - beta_2) * grad_w2 ** 2

    m4 = beta_1 * m4 + (1 - beta_1) * grad_b2
    v4 = beta_2 * v4 + (1 - beta_2) * grad_b2 ** 2

    D_W1 = D_W1 - (learing_rate / (np.sqrt(v1 /(1-beta_2) ) + eps)) * (m1/(1-beta_1))
    D_b1 = D_b1 - (learing_rate / (np.sqrt(v2 /(1-beta_2) ) + eps)) * (m2/(1-beta_1))
    
    D_W2 = D_W2 - (learing_rate / (np.sqrt(v3 /(1-beta_2) ) + eps)) * (m3/(1-beta_1))
    D_b2 = D_b2 - (learing_rate / (np.sqrt(v4 /(1-beta_2) ) + eps)) * (m4/(1-beta_1))

    # Func: Forward Feed for G
    Z = np.random.uniform(-1., 1., size=[1, G_input])
    Gl1 = Z.dot(G_W1) + G_b1
    Gl1A = arctan(Gl1)
    Gl2 = Gl1A.dot(G_W2) + G_b2
    current_fake_data = log(Gl2)

    Dl1 = current_fake_data.dot(D_W1) + D_b1
    Dl1_A = ReLu(Dl1)
    Dl2 = Dl1_A.dot(D_W2) + D_b2
    Dl2_A = log(Dl2)

    # Func: Cost G
    G_cost = -np.log(Dl2_A)

    # Func: Gradient
    grad_G_w2_part_1 = ((-1/Dl2_A) * d_log(Dl2).dot(D_W2.T) * (d_ReLu(Dl1))).dot(D_W1.T)
    grad_G_w2_part_2 = d_log(Gl2)
    grad_G_w2_part_3 = Gl1A
    grad_G_w2 = grad_G_w2_part_3.T.dot(grad_G_w2_part_1 * grad_G_w2_part_2)
    grad_G_b2 = grad_G_w2_part_1 * grad_G_w2_part_2

    grad_G_w1_part_1 = (grad_G_w2_part_1 * grad_G_w2_part_2).dot(G_W2.T)
    grad_G_w1_part_2 = d_arctan(Gl1)
    grad_G_w1_part_3 = Z
    grad_G_w1 = grad_G_w1_part_3.T.dot(grad_G_w1_part_1 * grad_G_w1_part_2)
    grad_G_b1 = grad_G_w1_part_1 * grad_G_w1_part_2

    # ---- Update Gradient ----
    m5 = beta_1 * m5 + (1 - beta_1) * grad_G_w1
    v5 = beta_2 * v5 + (1 - beta_2) * grad_G_w1 ** 2

    m6 = beta_1 * m6 + (1 - beta_1) * grad_G_b1
    v6 = beta_2 * v6 + (1 - beta_2) * grad_G_b1 ** 2

    m7 = beta_1 * m7 + (1 - beta_1) * grad_G_w2
    v7 = beta_2 * v7 + (1 - beta_2) * grad_G_w2 ** 2

    m8 = beta_1 * m8 + (1 - beta_1) * grad_G_b2
    v8 = beta_2 * v8 + (1 - beta_2) * grad_G_b2 ** 2

    G_W1 = G_W1 - (learing_rate / (np.sqrt(v5 /(1-beta_2) ) + eps)) * (m5/(1-beta_1))
    G_b1 = G_b1 - (learing_rate / (np.sqrt(v6 /(1-beta_2) ) + eps)) * (m6/(1-beta_1))
    
    G_W2 = G_W2 - (learing_rate / (np.sqrt(v7 /(1-beta_2) ) + eps)) * (m7/(1-beta_1))
    G_b2 = G_b2 - (learing_rate / (np.sqrt(v8 /(1-beta_2) ) + eps)) * (m8/(1-beta_1))

    # --- Print Error ----
    print("Current Iter: ",iter, " Current D cost:",D_cost, " Current G cost: ", G_cost,end='\r')
    

    # ---- Print to Out put ----
    if iter%1000 == 0:
        Z = np.random.uniform(-1., 1., size=[16, G_input])
        Gl1 = Z.dot(G_W1) + G_b1
        Gl1A = arctan(Gl1)
        Gl2 = Gl1A.dot(G_W2) + G_b2
        current_fake_data = log(Gl2)
    
        fig = plot(current_fake_data)
        plt.savefig('temp_arctan/{}.png'.format(str(iter).zfill(3)+"_"+str(G_input)+"_"+str(hidden_input)), bbox_inches='tight')
        plt.close(fig)



# -- end code --