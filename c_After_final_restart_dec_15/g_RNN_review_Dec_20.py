import numpy as np

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - tanh(x) ** 2

np.random.seed(1234)
x = np.array([
    [1,1,1,1],
    [1,1,0,1],
    [0,0,1,1],
    [1,1,0,0],
    [0,0,0,0],
    [1,1,1,0]
])

y = x.sum(axis=1)

wx = np.random.randn()
wrec = np.random.randn()
wx_learning = 0.001
wrec_learning = 0.0002

states = np.zeros((x.shape[0],x.shape[1]+1))
grad_overtime = np.zeros(x.shape)


print("Data :",x.shape)
print("states :",states.shape)
number_epoch = 13000

for iter in range(number_epoch):
    
    state = states[:,0] * wrec + x[:,0] * wx
    # state = tanh(state)
    states[:,1] = state

    state = states[:,1] * wrec + x[:,1] * wx
    # state = tanh(state)
    states[:,2] = state

    state = states[:,2] * wrec + x[:,2] * wx
    # state = tanh(state)
    states[:,3] = state

    state = states[:,3] * wrec + x[:,3] * wx
    # state = tanh(state)
    states[:,4] = state

    cost = np.square(states[:,4] - y).sum() / len(x)
    
    if iter % 1000 == 0 :
        print("Current Iter: ", iter, " current error: ",cost)

    grad_overtime[:,3] = (states[:,4] - np.squeeze(y)) * (2/len(x))
    grad_overtime[:,2] = grad_overtime[:,3] * wrec 
    grad_overtime[:,1] = grad_overtime[:,2] * wrec
    grad_overtime[:,0] = grad_overtime[:,1] * wrec

    grad_wx = np.sum(grad_overtime[:,3] * x[:,3] + 
                     grad_overtime[:,2] * x[:,2] + 
                     grad_overtime[:,1] * x[:,1]  + 
                     grad_overtime[:,0] * x[:,0])

    grad_rec = np.sum(grad_overtime[:,3] * states[:,3] + 
                      grad_overtime[:,2] * states[:,2] + 
                      grad_overtime[:,1] * states[:,1]  + 
                      grad_overtime[:,0] * states[:,0])
    
    wx = wx - wx_learning * grad_wx
    wrec = wrec - wrec_learning * grad_rec




state = states[:,0] * wrec + x[:,0] * wx
# state = tanh(state)
states[:,1] = state

state = states[:,1] * wrec + x[:,1] * wx
# state = tanh(state)
states[:,2] = state

state = states[:,2] * wrec + x[:,2] * wx
# state = tanh(state)
states[:,3] = state

state = states[:,3] * wrec + x[:,3] * wx
# state = tanh(state)
states[:,4] = state


print(y)

print(states[:,4])
print(np.round(states[:,4]))


# -- end code --