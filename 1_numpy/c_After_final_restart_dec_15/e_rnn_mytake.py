import numpy as np

np.random.seed(1234)

def log(x):
    return 1 / ( 1+ np.exp(-1*x))

def d_log(x):
    return log(x) * ( 1- log(x))

x = np.array([
    [1,0],
    [1,1]
])

y = np.array(
    [1,2],
)

wx = 0.5
wrec = 1.5

states = np.zeros((2,3))
    
# 1. Forward feed
current_x = x[:,0]
state_1 = states[:,0] * wrec + current_x * wx
states[:,1] = state_1

current_x = x[:,1]
state_2 = states[:,1] * wrec + current_x * wx
states[:,2] = state_2


print("All State: \n",states)



# 2. Back Propagation
grad_overtime = np.zeros((2,3))

grad_out_part_1 = 2 * (states[:,2] - y ) / 2
grad_overtime[:,2] = grad_out_part_1
grad_overtime[:,1] = grad_overtime[:,2] * wrec
grad_overtime[:,0] = grad_overtime[:,1] * wrec


print("All grad : \n",grad_overtime)


grad_wx = np.sum(grad_overtime[:,2] * x[:,1] + grad_overtime[:,1] * x[:,0])
grad_wrec= np.sum(grad_overtime[:,2] * states[:,1] + grad_overtime[:,1] * states[:,0])

print("Grad Wx:\n",grad_wx)
print("Grad Wrec:\n",grad_wrec)


# -- end code --