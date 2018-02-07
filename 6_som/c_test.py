import numpy as np,sys
from matplotlib import pyplot as plt
from matplotlib import patches as patches

np.random.seed(5678)

# 1. Declare data Set and hyper parameters
raw_data = np.random.randint(0, 255, (100, 3))
net = np.random.random((3,5,5))
n_iterations = 20000 
init_learning_rate = 0.003  

init_radius = max(net.shape[0], net.shape[1]) / 2
time_constant = n_iterations / np.log(init_radius)
raw_data = raw_data/raw_data.max()

# 2. Train n_iterations
for iter in range(n_iterations):
    
    rand_index = np.random.randint(0,len(raw_data))
    current_data = raw_data[rand_index,:]

    best_match_subtract = (net.T-current_data).T
    best_match = best_match_subtract ** 2 
    best_match_sum = best_match.sum(axis=0)
    bmu_idx = np.squeeze(np.asarray(np.where(best_match_sum == best_match_sum.min())))

    # print("Original Net :" ,net)
    # print("Original Data :" ,current_data)
    # print("Modifed Net :" ,best_match)
    # print("Total Sum: ", best_match_sum)
    # print("Print index : ", bmu_idx)

    # --------- ERRRO CATCHING --------
    # try:
    #      bmu_idx.shape[1]
    #      bmu_idx =np.array([bmu_idx[0,0],bmu_idx[1,0]]) 
    # except: 
    #      pass
    # --------- ERRRO CATCHING --------

    r = init_radius * np.exp(-iter / time_constant)
    l = init_learning_rate * np.exp(-iter / n_iterations)
    effect_matrix = np.zeros((net.shape[1],net.shape[2]))

    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
            if w_dist <= r**2:
                effect_matrix[x,y] = np.exp(-w_dist / (2* (r**2)))

    net = net - (l * np.multiply(effect_matrix,best_match_subtract))








fig = plt.figure()
# setup axes
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim((0, net.shape[1]+2))
ax.set_ylim((0, net.shape[2]+2))
ax.set_title('Self-Organising Map after %d iterations' % n_iterations)

# plot the rectangles
net = (net - net.min()) / (net.max() - net.min())

for x in range(1,net.shape[1]+1):
    for y in range(1,net.shape[2]+1):
        ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1, facecolor=net[:,x-1,y-1],edgecolor='none'))    
        plt.pause(0.1)
print(net)

plt.show()


# for x in range(1, net.shape[0] + 1):
#     for y in range(1, net.shape[1] + 1):
#         ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
#                      facecolor=net[x-1,y-1,:],
#                      edgecolor='none'))
# plt.show()

# -- end code --