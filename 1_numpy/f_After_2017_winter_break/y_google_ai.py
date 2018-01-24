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

num_epoch = 5000

wx_c = np.random.randn()
w_rec_c = np.random.randn()

wx_g = np.random.randn()
w_rec_g = np.random.randn()

lr_wx = 0.01
lr_wrec = 0.0001


for iter in range(num_epoch):

    c1 = x[:,0].dot(wx_c) + h[:,0].dot(w_rec_c)
    c1_act = tanh(c1)
    g1 = x[:,0].dot(wx_g) + h[:,0].dot(w_rec_g)
    g1_act = tanh(g1)
    h[:,1] = g1_act * h[:,0] + (1-g1_act) * c1_act

    c2 = x[:,1].dot(wx_c) + h[:,1].dot(w_rec_c)
    c2_act = tanh(c2)
    g2 = x[:,1].dot(wx_g) + h[:,1].dot(w_rec_g)
    g2_act = tanh(g2)
    h[:,2] = g2_act * h[:,1] + (1-g2_act) * c2_act

    c3 = x[:,2].dot(wx_c) + h[:,2].dot(w_rec_c)
    c3_act = tanh(c3)
    g3 = x[:,2].dot(wx_g) + h[:,2].dot(w_rec_g)
    g3_act = tanh(g3)
    h[:,3] = g3_act * h[:,2] + (1-g3_act) * c3_act

    cost = np.square(h[:,3] - np.squeeze(y)).sum() * 0.5

    if iter %100 == 0 :
        print("current iter : ",iter, " current cost: ",cost,end='\r')

    grad_3_common_c = (h[:,3] - np.squeeze(y)) * ( 1-g3_act ) * d_tanh(c3)
    grad_2_common_c = grad_3_common_c * (w_rec_c) * (1-g2_act) * d_tanh(c2) 
    grad_1_common_c = grad_2_common_c * (w_rec_c) * (1-g1_act) * d_tanh(c1) 
    
    grad_3_common_g = (h[:,3] - np.squeeze(y))    * ( h[:,2]-g3_act ) * d_tanh(g3)
    grad_2_common_g = grad_3_common_g * (w_rec_g) * ( h[:,1]-g2_act ) * d_tanh(g2) 
    grad_1_common_g = grad_2_common_c * (w_rec_g) * ( h[:,0]-g1_act ) * d_tanh(g1) 

    grad_wx_c = np.sum(
        grad_3_common_c * x[:,2] + 
        grad_2_common_c * x[:,1] + 
        grad_1_common_c * x[:,0] 
    )
    grad_wrec_c = np.sum(
        grad_3_common_c * h[:,2] + 
        grad_2_common_c * h[:,1] + 
        grad_1_common_c * h[:,0] 
    )
    
    grad_wx_g = np.sum(
        grad_3_common_g * x[:,2] + 
        grad_2_common_g * x[:,1] + 
        grad_1_common_g * x[:,0] 
    )
    grad_wrec_g = np.sum(
        grad_3_common_g * h[:,2] + 
        grad_2_common_g * h[:,1] + 
        grad_1_common_g * h[:,0] 
    )
    
    wx_c = wx_c - lr_wx * grad_wx_c
    w_rec_c = w_rec_c - lr_wrec * grad_wrec_c

    wx_g = wx_g - lr_wx * grad_wx_g
    w_rec_g = w_rec_g - lr_wrec * grad_wrec_g
    
c1 = x[:,0].dot(wx_c) + h[:,0].dot(w_rec_c)
c1_act = tanh(c1)
g1 = x[:,0].dot(wx_g) + h[:,0].dot(w_rec_g)
g1_act = tanh(g1)
h[:,1] = g1_act * h[:,0] + (1-g1_act) * c1_act

c2 = x[:,1].dot(wx_c) + h[:,1].dot(w_rec_c)
c2_act = tanh(c2)
g2 = x[:,1].dot(wx_g) + h[:,1].dot(w_rec_g)
g2_act = tanh(g2)
h[:,2] = g2_act * h[:,1] + (1-g2_act) * c2_act

c3 = x[:,2].dot(wx_c) + h[:,2].dot(w_rec_c)
c3_act = tanh(c3)
g3 = x[:,2].dot(wx_g) + h[:,2].dot(w_rec_g)
g3_act = tanh(g3)
h[:,3] = g3_act * h[:,2] + (1-g3_act) * c3_act

print('\n\n------------------')
print(h[:,3].T)
print('------------------')
print(y.T)
print('------------------')



# -----------------------------------------------------------------------

# -- end code --