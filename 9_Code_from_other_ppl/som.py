import numpy as np
raw_data = np.random.randint(0, 255, (3, 100))
data = raw_data


network_dimensions = np.array([5, 5])
n_iterations = 2000
init_learning_rate = 0.01
# establish size variables based on data
m = raw_data.shape[0]
n = raw_data.shape[1]

# weight matrix (i.e. the SOM) needs to be one m-dimensional vector for each neuron in the SOM
net = np.random.random((network_dimensions[0], network_dimensions[1], m))

# initial neighbourhood radius
init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
# radius decay parameter
time_constant = n_iterations / np.log(init_radius)
r = 0.01

def find_bmu(t, net, m):
    """
        Find the best matching unit for a given vector, t, in the SOM
        Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional BMU
                 and bmu_idx is the index of this vector in the SOM
    """
    bmu_idx = np.array([0, 0])
    # set the initial minimum distance to a huge number
    min_dist = np.iinfo(np.int).max
    # calculate the high-dimensional distance between each neuron and the input
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
            sq_dist = np.sum((w - t) ** 2)
            if sq_dist < min_dist:
                min_dist = sq_dist
                bmu_idx = np.array([x, y])
    # get vector corresponding to bmu_idx
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)

data = raw_data / data.max()

def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))

# select a training example at random
t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))

# find its Best Matching Unit
bmu, bmu_idx = find_bmu(t, net, m)

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

# now we know the BMU, update its weight vector to move closer to input
# and move its neighbours in 2-D space closer
# by a factor proportional to their 2-D distance from the BMU
for x in range(net.shape[0]):
    for y in range(net.shape[1]):

        w = net[x, y, :].reshape(m, 1)
        # get the 2-D distance (again, not the actual Euclidean distance)
        w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
        # if the distance is within the current neighbourhood radius
        if w_dist <= r**2:
            # calculate the degree of influence (based on the 2-D distance)
            influence = calculate_influence(w_dist, r)
            # now update the neuron's weight using the formula:
            # new w = old w + (learning rate * influence * delta)
            # where delta = input vector (t) - old w
            new_w = w + (l * influence * (t - w))
            # commit the new weight
            net[x, y, :] = new_w.reshape(1, 3)

# -- end code --