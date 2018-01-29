import numpy as np

np.random.seed(56789)

def ReLU(x):
    mask = (x > 0) * 1.0
    return x * mask
def d_ReLU(x):
    mask = (x > 0) * 1.0
    return mask

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(X) ** 2

def log(x):
    return 1 / (1 + np.exp(-1 *x))
def d_log(x):
    return log(x) * (1 - log(x))

# 1. Declare training data and hyper parameter
num_epoch = 1
learing_rate = 0.001

x = np.array([
    [0,0.5,0],
    [0.4,-0.4,0.3],
    [0.1,0.3,0.6]    
])

y = np.array([
    [0,0.5,0.5],
    [0.4,0,0.3],
    [0.1,0.4,1.0]    
])

h = np.zeros((x.shape[0],x.shape[1] + 1))

wrecyy,wxyy = np.random.randn() * 0.2,np.random.randn() * 0.2
wrechh,wxhh = np.random.randn() * 0.2,np.random.randn() * 0.2
wrecgy,wxgy = np.random.randn() * 0.2,np.random.randn() * 0.2
wrecgh,wxgh = np.random.randn() * 0.2,np.random.randn() * 0.2


for iter in range(num_epoch):

    # ---- Forward Feed at TS = 1 -------
    yy1 = wrecyy * h[:,0] + wxyy * x[:,0]
    yy1A = ReLU(yy1)

    hh1 = wrechh * h[:,0] + wxhh * x[:,0]
    hh1A = tanh(hh1)

    gy1 = wrecgy * h[:,0] + wxgy * x[:,0]
    gy1A = log(gy1)

    gh1 = wrecgh * h[:,0] + wxgh * x[:,0]
    gh1A = log(gh1)

    y1 = gy1A * x[:,0] + ( 1-gy1A ) * yy1A
    h[:,1] = gh1A * h[:,0] + ( 1-gh1A ) * hh1A
    
    # ---- Forward Feed at TS = 2 -------
    yy2 = wrecyy * h[:,1] + wxyy * x[:,1]
    yy2A = ReLU(yy2)

    hh2 = wrechh * h[:,1] + wxhh * x[:,1]
    hh2A = tanh(hh2)

    gy2 = wrecgy * h[:,1] + wxgy * x[:,1]
    gy2A = log(gy2)

    gh2 = wrecgh * h[:,1] + wxgh * x[:,1]
    gh2A = log(gh2)

    y2 = gy2A * x[:,1] + ( 1-gy2A ) * yy2A
    h[:,2] = gh2A * h[:,1] + ( 1-gh2A ) * hh2A

    # ---- Forward Feed at TS = 3 -------
    yy3 = wrecyy * h[:,2] + wxyy * x[:,2]
    yy3A = ReLU(yy3)

    hh3 = wrechh * h[:,2] + wxhh * x[:,2]
    hh3A = tanh(hh3)

    gy3 = wrecgy * h[:,2] + wxgy * x[:,2]
    gy3A = log(gy3)

    gh3 = wrecgh * h[:,2] + wxgh * x[:,2]
    gh3A = log(gh3)

    y3 = gy3A * x[:,2] + ( 1-gy3A ) * yy3A
    h[:,3] = gh3A * h[:,2] + ( 1-gh3A ) * hh3A

    cost_y1,cost_h1 = np.square(y1-y[:,0]).sum() * 0.5,np.square(h[:,1]-y[:,0] ).sum() * 0.5
    cost_y2,cost_h2 = np.square(y2-y[:,1]).sum() * 0.5,np.square(h[:,2]-y[:,1] ).sum() * 0.5
    cost_y3,cost_h3 = np.square(y3-y[:,2]).sum() * 0.5,np.square(h[:,3]-y[:,2] ).sum() * 0.5

    
            












# ---- end code ---   
