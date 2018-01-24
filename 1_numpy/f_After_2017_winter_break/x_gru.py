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

wx_r = np.random.randn()
w_rec_r = np.random.randn()

wx_u = np.random.randn()
w_rec_u = np.random.randn()

wx_c = np.random.randn()
w_rec_c = np.random.randn()


lr_wx = 0.01
lr_wrec = 0.0001


for iter in range(num_epoch):

    r1 = x[:,0].dot(wx_r) + h[:,0].dot(w_rec_r)
    r1_act = log(r1)
    u1 = x[:,0].dot(wx_u) + h[:,0].dot(w_rec_u)
    u1_act = log(u1)
    c1 = x[:,0].dot(wx_c) + (r1*h[:,0]).dot(w_rec_c)
    c1_act = tanh(c1)

    h[:,1] = u1_act * h[:,0] + (1-u1_act) * c1_act

    r2 = x[:,1].dot(wx_r) + h[:,1].dot(w_rec_r)
    r2_act = log(r2)
    u2 = x[:,1].dot(wx_u) + h[:,1].dot(w_rec_u)
    u2_act = log(u2)
    c2 = x[:,1].dot(wx_c) + (r2*h[:,1]).dot(w_rec_c)
    c2_act = tanh(c2)

    h[:,2] = u2_act * h[:,1] + (1-u2_act) * c2_act

    r3 = x[:,2].dot(wx_r) + h[:,2].dot(w_rec_r)
    r3_act = log(r3)
    u3 = x[:,2].dot(wx_u) + h[:,2].dot(w_rec_u)
    u3_act = log(u3)
    c3 = x[:,2].dot(wx_c) + (r3*h[:,2]).dot(w_rec_c)
    c3_act = tanh(c3)

    h[:,3] = u3_act * h[:,2] + (1-u3_act) * c3_act

    cost = np.square(h[:,3] - np.squeeze(y)).sum() * 0.5
    if iter %100 == 0 :
        print("current iter : ",iter, " current cost: ",cost,end='\r')

    grad_3_common_c = (h[:,3] - np.squeeze(y))    * (1-u3_act) * d_tanh(c3)
    grad_2_common_c = grad_3_common_c * (w_rec_c) * (
                    (r3) * (1-u2_act) * d_tanh(u2) + 
                    (h[:,2]) * (d_log(r3)) * (w_rec_r) * (1-u2_act) * d_tanh(u2)
                    )
    grad_1_common_c = grad_2_common_c * (w_rec_c) * (
                    (r1) * (1-u1_act) * d_tanh(u1) + 
                    (h[:,1]) * (d_log(r2)) * (w_rec_r) * (1-u1_act) * d_tanh(u1)
                    )









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

    wx_c = wx_c  - lr_wx * grad_wx_c
    w_rec_c = w_rec_c  - lr_wrec * grad_wrec_c
    
    


# -----------------------------------------------------------------------

# -- end code --