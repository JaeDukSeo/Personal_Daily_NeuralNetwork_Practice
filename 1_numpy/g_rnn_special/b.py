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

hidden_states = np.zeros((x.shape[0],x.shape[1] + 1))
gradients = np.zeros((x.shape))

num_epoch = 10000

wx = (np.random.randn() * 0.2) - 0.1
w_rec = (np.random.randn() * 0.2) - 0.1

lr_wx = 0.1
lr_wrec = 0.09


for iter in range(num_epoch):
    
    layer_1 = x[:,0].dot(wx) + hidden_states[:,0].dot(w_rec)
    layer_1_act = log(layer_1)
    hidden_states[:,1] = layer_1_act

    layer_2 = x[:,1].dot(wx) + hidden_states[:,1].dot(w_rec)
    layer_2_act = log(layer_2)
    hidden_states[:,2] = layer_1_act

    layer_3 = x[:,2].dot(wx) + hidden_states[:,2].dot(w_rec)
    layer_3_act = tanh(layer_3)
    hidden_states[:,3] = layer_3_act

    cost = np.square(layer_3_act - np.squeeze(y)).sum() * 0.5
    
    if iter %100 == 0 :
        print("current iter : ",iter, " current cost: ",cost,end='\r')

    grad_common_3 = (layer_3_act - np.squeeze(y)) * d_log(layer_3)
    grad_common_2 = grad_common_3 * w_rec * d_log(layer_2)   
    grad_common_1 = grad_common_2 * w_rec * d_log(layer_1)

    grad_wx = np.sum(
        grad_common_3 * x[:,2] + 
        grad_common_2 * x[:,1] + 
        grad_common_1 * x[:,0] 
    )
    
    grad_w_rec = np.sum(
        grad_common_3 * hidden_states[:,2] + 
        grad_common_2 * hidden_states[:,1] + 
        grad_common_1 * hidden_states[:,0] 
    )  

    wx = wx - lr_wx * grad_wx
    w_rec = w_rec - lr_wx * grad_w_rec
    

layer_1 = x[:,0].dot(wx) + hidden_states[:,0].dot(w_rec)
layer_1_act = log(layer_1)
hidden_states[:,1] = layer_1_act

layer_2 = x[:,1].dot(wx) + hidden_states[:,1].dot(w_rec)
layer_2_act = log(layer_2)
hidden_states[:,2] = layer_1_act

layer_3 = x[:,2].dot(wx) + hidden_states[:,2].dot(w_rec)
layer_3_act = tanh(layer_3)


print('\n\n------------------')
print(layer_3_act.T)
print('------------------')
print(y.T)
print('------------------')

# -- end code --