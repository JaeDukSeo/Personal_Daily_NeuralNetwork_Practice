import numpy as np

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

x = np.array([
    [-0.2,-0.2,-0.6],
    [-0.3,-0.2,0],
    [0,0,0],
    [0.1,0.1,0.3],
    [0,0.5,0.5]
])

y = np.array([
    [-1.0],
    [-0.5],
    [0],
    [0.5],
    [1.0]    
])

h = np.zeros((x.shape[0],x.shape[1] + 1))
gradients = np.zeros((x.shape))

num_epoch = 10000

wx_c = (np.random.randn() * 0.2) - 0.1
w_rec_c = (np.random.randn() * 0.2) - 0.1

wx_g = (np.random.randn() * 0.2) - 0.1
w_rec_g = (np.random.randn() * 0.2) - 0.1

lr_wx_c = 0.1
lr_wrec_c = 0.09
lr_wx_g = 0.1
lr_wrec_g = 0.09

for iter in range(num_epoch):
    
    c1 = x[:,0].dot(wx_c) + h[:,0].dot(w_rec_c)
    c1A = tanh(c1)
    g1 = x[:,0].dot(wx_g) + h[:,0].dot(w_rec_g)
    g1A = tanh(g1)
    h[:,1] = g1A * h[:,0] + (1-g1A) * c1A

    c2 = x[:,1].dot(wx_c) + h[:,1].dot(w_rec_c)
    c2A = tanh(c2)
    g2 = x[:,1].dot(wx_g) + h[:,1].dot(w_rec_g)
    g2A = tanh(g2)
    h[:,2] = g2A * h[:,1] + (1-g2A) * c2A

    c3 = x[:,2].dot(wx_c) + h[:,2].dot(w_rec_c)
    c3A = tanh(c3)
    g3 = x[:,2].dot(wx_g) + h[:,2].dot(w_rec_g)
    g3A = tanh(g3)
    h[:,3] = g3A * h[:,2] + (1-g3A) * c3A

    cost = np.square(h[:,3] - np.squeeze(y)).sum() * 0.5

    if iter %100 == 0 :
        print("current iter : ",iter, " current cost: ",cost,end='\r')

    # ---------------------------------------------------------------
    grad_3_common_c = (h[:,3] - np.squeeze(y)) * (1 - g3A) * (d_tanh(c3))
    grad_3_common_g = (h[:,3] - np.squeeze(y)) * (h[:,2] - c3A) * (d_tanh(g3))
    # ---------------------------------------------------------------


    # ---------------------------------------------------------------
    grad_2_common_1 = grad_3_common_g * (w_rec_g)
    grad_2_common_2 = grad_3_common_c * (w_rec_c)
    grad_2_common_h = (h[:,3] - np.squeeze(y)) * (g3A)
    
    grad_2_c_end = (1-g2A) * (d_tanh(c2))
    grad_2_g_end = (h[:,1] - c2A)* (d_tanh(g2))
    
    grad_2_common_c = (grad_2_common_1 + grad_2_common_2 + grad_2_common_h) * grad_2_c_end
    grad_2_common_g = (grad_2_common_1 + grad_2_common_2 + grad_2_common_h) * grad_2_g_end
    # ---------------------------------------------------------------


    # ---------------------------------------------------------------
    grad_1_common_1 = grad_2_common_g * (w_rec_g)
    grad_1_common_2 = grad_2_common_c * (w_rec_c)
    grad_1_common_h = (grad_2_common_1 + grad_2_common_2 + grad_2_common_h) * (g2A)
    
    grad_1_c_end = (1-g1A) * (d_tanh(c1))
    grad_1_g_end = (h[:,0] - c2A)* (d_tanh(g1))

    grad_1_common_c = (grad_1_common_1 + grad_1_common_2 + grad_1_common_h) * grad_1_c_end
    grad_1_common_g = (grad_1_common_1 + grad_1_common_2 + grad_1_common_h) * grad_1_g_end
    # ---------------------------------------------------------------

    grad_wx_c = np.sum(
        grad_3_common_c * x[:,2] + 
        grad_2_common_c * x[:,1] + 
        grad_1_common_c * x[:,0] 
    )    

    grad_w_rec_c = np.sum(
        grad_3_common_c * h[:,2] + 
        grad_2_common_c * h[:,1] + 
        grad_1_common_c * h[:,0] 
    )    

    grad_wx_g = np.sum(
        grad_3_common_g * x[:,2] + 
        grad_2_common_g * x[:,1] + 
        grad_1_common_g * x[:,0] 
    )    

    grad_w_rec_g = np.sum(
        grad_3_common_g * h[:,2] + 
        grad_2_common_g * h[:,1] + 
        grad_1_common_g * h[:,0] 
    )   

    wx_c = wx_c - lr_wx_c*grad_wx_c
    w_rec_c  = w_rec_c - lr_wrec_c*grad_w_rec_c

    wx_g = wx_g - lr_wx_g*grad_wx_g
    w_rec_g  = w_rec_g - lr_wrec_g*grad_w_rec_g
    

c1 = x[:,0].dot(wx_c) + h[:,0].dot(w_rec_c)
c1A = tanh(c1)
g1 = x[:,0].dot(wx_g) + h[:,0].dot(w_rec_g)
g1A = tanh(g1)
h[:,1] = g1A * h[:,0] + (1-g1A) * c1A

c2 = x[:,1].dot(wx_c) + h[:,1].dot(w_rec_c)
c2A = tanh(c2)
g2 = x[:,1].dot(wx_g) + h[:,1].dot(w_rec_g)
g2A = tanh(g2)
h[:,2] = g2A * h[:,1] + (1-g2A) * c2A

c3 = x[:,2].dot(wx_c) + h[:,2].dot(w_rec_c)
c3A = tanh(c3)
g3 = x[:,2].dot(wx_g) + h[:,2].dot(w_rec_g)
g3A = tanh(g3)
h[:,3] = g3A * h[:,2] + (1-g3A) * c3A
print('\n\n------------------')
print(h[:,3].T)
print('------------------')
print(y.T)
print('------------------')

# -- end code --