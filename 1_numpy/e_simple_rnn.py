import numpy as np

np.random.seed(1234)

def log(x):
    return 1 / (1 + np.exp(-1 *x))

def d_log(x):
    return log(x) * (1 - log(x))


x = np.array([
    [0,0,0],
    [0,0,1],
    [1,1,1]
])

y = np.array([
    [0],
    [1],
    [3]
])

wrec = 0.05
wx = 0.08

states = np.zeros((x.shape[0],x.shape[1] + 1))
gradients = np.zeros((x.shape))

epoch = 100000

for iter in range(epoch):

    state_1_in = x[:,0].dot(wx) + states[:,0].dot(wrec)
    state_1_out = log(state_1_in)
    states[:,1] = state_1_out

    state_2_in = x[:,1].dot(wx) + states[:,1].dot(wrec)
    state_2_out = log(state_2_in)
    states[:,2] = state_2_out

    state_3_in = x[:,2].dot(wx) + states[:,2].dot(wrec)
    states[:,3] = state_3_in

    cost = np.square(states[:,3]  - np.squeeze(y)) / 2

    if iter%1000 ==0:
        print("current iter: ", iter, " currnet cost:", cost)

    gradients[:,2] = states[:,3]  - np.squeeze(y)
    gradients[:,1] = gradients[:,2].dot(wrec) * d_log(state_2_in)
    gradients[:,0] = gradients[:,1].dot(wrec) * d_log(state_1_in)

    grad_wx = np.sum(
        gradients[:,2] * x[:,2] +
        gradients[:,1] * x[:,1] +
        gradients[:,0] * x[:,0]         
    )

    grad_wrec = np.sum(gradients[:,2]*states[:,2]+
                gradients[:,1]*states[:,1]+
                gradients[:,0]*states[:,0])

    wx = wx - 0.01 * grad_wx
    wrec = wrec - 0.0001 * grad_wrec
    


state_1_in = x[:,0].dot(wx) + states[:,0].dot(wrec)
state_1_out = log(state_1_in)
states[:,1] = state_1_out

state_2_in = x[:,1].dot(wx) + states[:,1].dot(wrec)
state_2_out = log(state_2_in)
states[:,2] = state_2_out

state_3_in = x[:,2].dot(wx) + states[:,2].dot(wrec)
states[:,3] = state_3_in

print(state_3_in)
print(np.round(state_3_in))



#  -- end code --