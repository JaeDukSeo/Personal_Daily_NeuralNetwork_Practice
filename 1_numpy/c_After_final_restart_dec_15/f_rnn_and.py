import numpy as np

x = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])

y = np.array([
    [0],
    [0],
    [0],
    [1]
])

wx = [1]
wrec = [0.2]

states = np.zeros((4,3))
grad_over_time =  np.zeros((4,3))

for iter in range(10):
    layer_1 = states[:,0] * wrec +  x[:,0] * wx 
    states[:,1] = layer_1

    layer_2 = states[:,1] * wrec +  x[:,1] * wx 
    states[:,2] = layer_2

    grad_2_part_3 = (states[:,2] - np.squeeze(y)) * 2 / 4
    grad_over_time[:,2] = grad_2_part_3
    grad_over_time[:,1] = grad_over_time[:,2] * wrec
    grad_over_time[:,0] = grad_over_time[:,1] * wrec
    
    grad_wx = np.sum(grad_over_time[:,2] * x[:,1] + grad_over_time[:,1] + x[:,0])
    grad_wrec = np.sum(grad_over_time[:,2] * states[:,1] + grad_over_time[:,1] + states[:,0])

    wx = wx - grad_wx
    wrec = wrec - 0.001 *grad_wrec


layer_1 = states[:,0] * wrec +  x[:,0] * wx 
states[:,1] = layer_1

layer_2 = states[:,1] * wrec +  x[:,1] * wx 
states[:,2] = layer_2 
    
print(layer_2)

# -- end code --