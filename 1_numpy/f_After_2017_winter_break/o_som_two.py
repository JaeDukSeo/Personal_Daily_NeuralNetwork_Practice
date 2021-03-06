import numpy as np,sys
from matplotlib import pyplot as plt
from matplotlib import patches as patches



np.random.seed(6789)

raw_data = np.random.randint(0, 255, (3, 100))
network_dimensions = np.array([4, 4])
n_iterations = 1
init_learning_rate = 0.01

normalise_data = True
normalise_by_column = False

m = raw_data.shape[0]
n = raw_data.shape[1]

# initial neighbourhood radius || radius decay parameter
init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
time_constant = n_iterations / np.log(init_radius)

data = raw_data
if normalise_data:
    if normalise_by_column:
        # normalise along each column
        col_maxes = raw_data.max(axis=0)
        data = raw_data / col_maxes[np.newaxis, :]
    else:
        # normalise entire dataset
        data = raw_data / data.max()

net = np.random.random((network_dimensions[0], network_dimensions[1], m))
start_net = np.array(net,copy=True)

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
            sq_dist = np.sqrt(np.sum((w - t) ** 2))
            if sq_dist < min_dist:
                min_dist = sq_dist
                bmu_idx = np.array([x, y])
    # get vector corresponding to bmu_idx
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)
def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))

array_dist = np.zeros((4,4))
array_influence = np.zeros((4,4))
for i in range(n_iterations):
    t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))
    
    # bmu, bmu_idx = find_bmu(t, net, m)
    Euclidean_distance = np.sqrt(np.subtract(net , np.squeeze(t)) ** 2)
    Euclidean_distance_sum = Euclidean_distance.sum(axis=2)
    bmu_idx = np.squeeze(np.asarray(np.where(Euclidean_distance_sum == Euclidean_distance_sum.min())))
    bmu = net[bmu_idx[0],bmu_idx[1],:].reshape(m, 1)

    # r = decay_radius(init_radius, i, time_constant)
    r = init_radius * np.exp(-i / time_constant)
    # l = decay_learning_rate(init_learning_rate, i, n_iterations)
    l = init_learning_rate * np.exp(-i / n_iterations)
    
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
            array_dist[x,y] = w_dist
            if w_dist <= r**2:
                influence = calculate_influence(w_dist, r)
                array_influence[x,y] = influence
                new_w = w + (l * influence * (t - w))
                net[x, y, :] = new_w.reshape(1, 3)
                

print('--------Sum of Smallest------------')
print(start_net[2,1,:].sum())
print('--------Start of the NetWork------------')
print(start_net.sum(axis=2))
print('--------Start of the Net * L------------')
print(l*start_net.sum(axis=2))


print('--------Difference ------------')
print(((start_net- net)**2).sum(axis=2))

print('--------Distance------------')
print(array_dist)
print('--------influence------------')
print(array_influence)


print('\n\n')
print('------- Start ---------')
print(start_net.)

print('------- expand ---------')
grad_learning = l * start_net
temp = np.stack((array_influence,array_influence,array_influence))


sys.exit()

final = start_net + (l * start_net * temp)
print('--------Made final ------------')
print(final.sum(axis=2))



print('-------Final Product-------------')
print(net.sum(axis=2))






























sys.exit()
fig = plt.figure()
# setup axes
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim((0, net.shape[0]+1))
ax.set_ylim((0, net.shape[1]+1))
ax.set_title('Self-Organising Map after %d iterations' % n_iterations)

# plot the rectangles
for x in range(1, net.shape[0] + 1):
    for y in range(1, net.shape[1] + 1):
        ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                     facecolor=net[x-1,y-1,:],
                     edgecolor='none'))
plt.show()





# -- end code --