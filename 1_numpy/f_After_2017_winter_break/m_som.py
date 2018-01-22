import numpy as np
from sklearn import datasets


np.random.seed(56789876)

digits = datasets.load_digits()
image = digits.images.T

image = np.reshape(image,(64,-1))

Grid_2D_net = np.random.randn(50,50,64)

num_epoch= 500
init_radius = 10
init_learning_rate = 0.1
time_constant = 3

for i in range(num_epoch):
        
    image_index = np.random.randint(0, len(image[1]))
    image_vector = image[:,image_index]

    Euclidean_distance = np.sqrt(np.subtract(Grid_2D_net , image_vector) ** 2)
    Euclidean_distance_sum = Euclidean_distance.sum(axis=2)
    bmu_idx = np.squeeze(np.asarray(np.where(Euclidean_distance_sum == Euclidean_distance_sum.min())))
    bmu = Grid_2D_net[bmu_idx[0],bmu_idx[1],:]
    
    r = init_radius * np.exp(-i / time_constant)
    l = init_learning_rate * np.exp(-i / num_epoch)

    for x in range(Grid_2D_net.shape[0]):
        for y in range(Grid_2D_net.shape[1]):
            w = Grid_2D_net[x, y, :]
            w_dist = np.sum(np.sqrt((bmu_idx - bmu_idx) ** 2))
            if w_dist <= r**2:
                influence = np.exp(-w_dist / (2* (r**2)))
                new_w = w + (l * influence * (image_vector - w))
                Grid_2D_net[x, y, :] = new_w

# -- end code --