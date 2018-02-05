import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches

np.random.seed(5678)

# 1. Declare data Set and hyper parameters
raw_data = np.random.randint(0, 255, (100, 3))
net = np.random.randn(3,5,5)
# net = np.zeros((3,5,5))   
n_iterations = 1
init_learning_rate = 0.01


# 2. Train
for iter in range(n_iterations):
    
    rand_index = np.random.randint(0,len(raw_data))
    current_data = raw_data[rand_index,:]

    best_match_subtract = (net.T-current_data).T
    best_match = np.sqrt(  best_match_subtract ** 2 )
    best_match_sum = best_match.sum(axis=0)
    bmu_idx = np.squeeze(np.asarray(np.where(best_match_sum == best_match_sum.min())))

    print("Original Net :" ,net)
    print("Original Data :" ,current_data)
    print("Modifed Net :" ,best_match)
    print("Total Sum: ", best_match_sum)
    print("Print index : ", bmu_idx)

    # --------- ERRRO CATCHING --------
    try:
         bmu_idx.shape[1]
         bmu_idx =np.array([bmu_idx[0,0],bmu_idx[1,0]]) 
    except: 
         pass
    # --------- ERRRO CATCHING --------

    r = 3 * np.exp(-iter / (n_iterations / np.log(5)))
    l = init_learning_rate * np.exp(-iter / n_iterations)
    effect_matrix = np.zeros((5,5))

    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
            if w_dist <= r**2:
                effect_matrix[x,y] = np.exp(-w_dist / (2* (r**2)))

    net = net - (l * np.multiply(effect_matrix,best_match_subtract))

# -- end code --