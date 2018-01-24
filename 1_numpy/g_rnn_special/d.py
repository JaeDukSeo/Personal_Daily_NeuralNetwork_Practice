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


# 0. Declare Training Data and Labels
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

# 1. Declare the Hidden States and weights and hyper parameters
h = np.zeros((x.shape[0],x.shape[1] + 1))
num_epoch = 10000

wx = (np.random.randn() * 0.2) - 0.1
w_rec = (np.random.randn() * 0.2) - 0.1

wx_i = 1
w_rec_i = 1

wx_c = wx
w_rec_c = w_rec

wx_g = wx
w_rec_g = w_rec

# 2. Learning rate for each networks
lr_wx = 0.1
lr_wrec = 0.09

lr_wx_c = 0.1
lr_wrec_c = 0.09
lr_wx_g = 0.1
lr_wrec_g = 0.09

# ------------ Updated Gate RNN Train -------
for iter in range(num_epoch):
    
    c1 = x[:,0].dot(wx_c) + h[:,0].dot(w_rec_c)
    c1A = tanh(c1)
    g1 = x[:,0].dot(wx_g) + h[:,0].dot(w_rec_g)
    g1A = log(g1)
    h[:,1] = g1A * h[:,0] + (1-g1A) * c1A

    c2 = x[:,1].dot(wx_c) + h[:,1].dot(w_rec_c)
    c2A = tanh(c2)
    g2 = x[:,1].dot(wx_g) + h[:,1].dot(w_rec_g)
    g2A = log(g2)
    h[:,2] = g2A * h[:,1] + (1-g2A) * c2A

    c3 = x[:,2].dot(wx_c) + h[:,2].dot(w_rec_c)
    c3A = tanh(c3)
    g3 = x[:,2].dot(wx_g) + h[:,2].dot(w_rec_g)
    g3A = log(g3)
    h[:,3] = g3A * h[:,2] + (1-g3A) * c3A

    cost = np.square(h[:,3] - np.squeeze(y)).sum() * 0.5

    # if iter %100 == 0 :
    #     print("current iter : ",iter, " current cost: ",cost,end='\r')

    # ---------------------------------------------------------------
    grad_3_common_c = (h[:,3] - np.squeeze(y)) * (1 - g3A) * (d_tanh(c3))
    grad_3_common_g = (h[:,3] - np.squeeze(y)) * (h[:,2] - c3A) * (d_log(g3))
    # ---------------------------------------------------------------


    # ---------------------------------------------------------------
    grad_2_common_1 = grad_3_common_g * (w_rec_g)
    grad_2_common_2 = grad_3_common_c * (w_rec_c)
    grad_2_common_h = (h[:,3] - np.squeeze(y)) * (g3A)
    
    grad_2_c_end = (1-g2A) * (d_tanh(c2))
    grad_2_g_end = (h[:,1] - c2A)* (d_log(g2))
    
    grad_2_common_c = (grad_2_common_1 + grad_2_common_2 + grad_2_common_h) * grad_2_c_end
    grad_2_common_g = (grad_2_common_1 + grad_2_common_2 + grad_2_common_h) * grad_2_g_end
    # ---------------------------------------------------------------


    # ---------------------------------------------------------------
    grad_1_common_1 = grad_2_common_g * (w_rec_g)
    grad_1_common_2 = grad_2_common_c * (w_rec_c)
    grad_1_common_h = (grad_2_common_1 + grad_2_common_2 + grad_2_common_h) * (g2A)
    
    grad_1_c_end = (1-g1A) * (d_tanh(c1))
    grad_1_g_end = (h[:,0] - c2A)* (d_log(g1))

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
print('\n-------------------')
print("Update Gate Training Done Final Results")
c1 = x[:,0].dot(wx_c) + h[:,0].dot(w_rec_c)
c1A = tanh(c1)
g1 = x[:,0].dot(wx_g) + h[:,0].dot(w_rec_g)
g1A = log(g1)
h[:,1] = g1A * h[:,0] + (1-g1A) * c1A

c2 = x[:,1].dot(wx_c) + h[:,1].dot(w_rec_c)
c2A = tanh(c2)
g2 = x[:,1].dot(wx_g) + h[:,1].dot(w_rec_g)
g2A = log(g2)
h[:,2] = g2A * h[:,1] + (1-g2A) * c2A

c3 = x[:,2].dot(wx_c) + h[:,2].dot(w_rec_c)
c3A = tanh(c3)
g3 = x[:,2].dot(wx_g) + h[:,2].dot(w_rec_g)
g3A = log(g3)
h[:,3] = g3A * h[:,2] + (1-g3A) * c3A
cost = np.square(h[:,3] - np.squeeze(y)).sum() * 0.5
print("Cost : ",cost)
print("Results : ", h[:,3].T)
print("Ground Truth : ", y.T)
print('-------------------\n')
# ------------ Updated Gate RNN Train -------

# ------------ Normal Gate RNN Train -------
# NOTE: Reclare the hidden states
h = np.zeros((x.shape[0],x.shape[1] + 1))
for iter in range(num_epoch):
    
    layer_1 = x[:,0].dot(wx) + h[:,0].dot(w_rec)
    layer_1_act = tanh(layer_1)
    h[:,1] = layer_1_act

    layer_2 = x[:,1].dot(wx) + h[:,1].dot(w_rec)
    layer_2_act = tanh(layer_2)
    h[:,2] = layer_1_act

    layer_3 = x[:,2].dot(wx) + h[:,2].dot(w_rec)
    layer_3_act = tanh(layer_3)
    h[:,3] = layer_3_act

    cost = np.square(layer_3_act - np.squeeze(y)).sum() * 0.5
    
    # if iter %100 == 0 :
    #     print("current iter : ",iter, " current cost: ",cost,end='\r')

    grad_common_3 = (layer_3_act - np.squeeze(y)) * d_tanh(layer_3)
    grad_common_2 = grad_common_3 * w_rec * d_tanh(layer_2)   
    grad_common_1 = grad_common_2 * w_rec * d_tanh(layer_1)

    grad_wx = np.sum(
        grad_common_3 * x[:,2] + 
        grad_common_2 * x[:,1] + 
        grad_common_1 * x[:,0] 
    )
    
    grad_w_rec = np.sum(
        grad_common_3 * h[:,2] + 
        grad_common_2 * h[:,1] + 
        grad_common_1 * h[:,0] 
    )  

    wx = wx - lr_wx * grad_wx
    w_rec = w_rec - lr_wx * grad_w_rec

print('\n-------------------')
print("Normal RNN Training Done Final Results")
layer_1 = x[:,0].dot(wx) + h[:,0].dot(w_rec)
layer_1_act = tanh(layer_1)
h[:,1] = layer_1_act

layer_2 = x[:,1].dot(wx) + h[:,1].dot(w_rec)
layer_2_act = tanh(layer_2)
h[:,2] = layer_1_act

layer_3 = x[:,2].dot(wx) + h[:,2].dot(w_rec)
layer_3_act = tanh(layer_3)
h[:,3] = layer_3_act
cost = np.square(layer_3_act - np.squeeze(y)).sum() * 0.5
print("Cost : ",cost)
print("Results : ", h[:,3].T)
print("Ground Truth : ", y.T)
print('-------------------\n')
# ------------ Normal Gate RNN Train -------

# ------------ Initialize Recurrent Networks Train -------
# NOTE: Reclare the hidden states
h = np.zeros((x.shape[0],x.shape[1] + 1))
for iter in range(num_epoch):
    
    layer_1 = x[:,0].dot(wx_i) + h[:,0].dot(w_rec_i)
    layer_1_act = ReLu(layer_1)
    h[:,1] = layer_1_act

    layer_2 = x[:,1].dot(wx_i) + h[:,1].dot(w_rec_i)
    layer_2_act = ReLu(layer_2)
    h[:,2] = layer_1_act

    layer_3 = x[:,2].dot(wx_i) + h[:,2].dot(w_rec_i)
    layer_3_act = ReLu(layer_3)
    h[:,3] = layer_3_act

    cost = np.square(layer_3_act - np.squeeze(y)).sum() * 0.5
    
    # if iter %100 == 0 :
    #     print("current iter : ",iter, " current cost: ",cost,end='\r')

    grad_common_3 = (layer_3_act - np.squeeze(y)) * d_ReLu(layer_3)
    grad_common_2 = grad_common_3 * w_rec_i * d_ReLu(layer_2)   
    grad_common_1 = grad_common_2 * w_rec_i * d_ReLu(layer_1)

    grad_wx = np.sum(
        grad_common_3 * x[:,2] + 
        grad_common_2 * x[:,1] + 
        grad_common_1 * x[:,0] 
    )
    
    grad_w_rec = np.sum(
        grad_common_3 * h[:,2] + 
        grad_common_2 * h[:,1] + 
        grad_common_1 * h[:,0] 
    )  

    wx_i = wx_i - lr_wx * grad_wx
    w_rec_i = w_rec_i - lr_wx * grad_w_rec
print('\n-------------------')
print("Initialize Recurrent Networks Training Done Final Results")
layer_1 = x[:,0].dot(wx_i) + h[:,0].dot(w_rec_i)
layer_1_act = ReLu(layer_1)
h[:,1] = layer_1_act

layer_2 = x[:,1].dot(wx_i) + h[:,1].dot(w_rec_i)
layer_2_act = ReLu(layer_2)
h[:,2] = layer_1_act

layer_3 = x[:,2].dot(wx_i) + h[:,2].dot(w_rec_i)
layer_3_act = ReLu(layer_3)
h[:,3] = layer_3_act
cost = np.square(layer_3_act - np.squeeze(y)).sum() * 0.5
print("Cost : ",cost)
print("Results : ", h[:,3].T)
print("Ground Truth : ", y.T)
print('-------------------\n')

# ------------ Initialize Recurrent Networks Train -------




# -- end code --