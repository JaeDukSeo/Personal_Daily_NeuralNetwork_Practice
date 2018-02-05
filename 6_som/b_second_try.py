import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches

np.random.seed(5678)

# 1. Declare data Set and hyper parameters
raw_data = np.random.randint(0, 255, (100, 3))
net = np.random.randn(3,5,5)
n_iterations = 100
init_learning_rate = 0.01


# 2. Train
for iter in range(n_iterations):
    
    rand_index = np.random.randint(0,len(raw_data))
    print(rand_index)





# -- end code --