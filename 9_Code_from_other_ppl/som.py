import numpy as np,sys
import matplotlib.pyplot as plt
from matplotlib import patches as patches
np.random.seed(51673)

raw_data = np.random.randint(0, 255, (3, 100))

network_dimensions = np.array([20, 20])
n_iterations = 2000
init_learning_rate = 0.1
# establish size variables based on data
m = raw_data.shape[0]
n = raw_data.shape[1]

# weight matrix (i.e. the SOM) needs to be one m-dimensional vector for each neuron in the SOM
net = np.random.random((network_dimensions[0], network_dimensions[1],m))

# initial neighbourhood radius
init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
# radius decay parameter
time_constant = n_iterations / np.log(init_radius)


normalise_data = True

# if True, assume all data is on common scale
# if False, normalise to [0 1] range along each column
normalise_by_column = False

# we want to keep a copy of the raw data for later
data = raw_data

# check if data needs to be normalised
if normalise_data:
    if normalise_by_column:
        # normalise along each column
        col_maxes = raw_data.max(axis=0)
        data = raw_data / col_maxes[np.newaxis, :]
    else:
        # normalise entire dataset
        data = raw_data / data.max()

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


# ---- radius and learning rate ------
def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)
# ---- radius and learning rate ------

# --------- what is this?? -----
def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))
# --------- what is this?? -----


# ---- Training Aspect -----
for i in range(n_iterations):
    
    # select a training example at random
    t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))

    temp = np.subtract(net,np.squeeze(t)) ** 2
    temp_sum = temp.sum(axis=2)
    bmu_idx_2 = np.squeeze(np.asarray(np.where(temp_sum == temp_sum.min())))
    bmu_2 = net[bmu_idx_2[0],bmu_idx_2[1],:].reshape(np.array([m, 1]))

    # find its Best Matching Unit
    # bmu, bmu_idx = find_bmu(t, net, m)

    bmu  = bmu_2
    bmu_idx  = bmu_idx_2

    # print("My : ",bmu_idx_2,bmu_2)
    # print("\noG : ",bmu_idx,bmu)
    # print('\n')
    # print(bmu_2-bmu)
    # print('\n')
    # print(bmu_idx_2-bmu_idx)
    # tsss = input()

    # decay the SOM parameters
    r = decay_radius(init_radius, i, time_constant)
    l = decay_learning_rate(init_learning_rate, i, n_iterations)

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


print('\n-----')
print(raw_data.shape)
print(net.shape)
print('\n-----')


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






sys.exit()
print("----- explore --------")
t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))
bmu, bmu_idx = find_bmu(t, net, m)
print(bmu,'\n',bmu_idx)
print("----ssss---------")

print(net.shape)
print(m)
tempw = net[0, 0, :].reshape(m, 1)
print(tempw.shape)
print(tempw - t)
q_dist = np.sum((tempw - t) ** 2)
print(q_dist)

sstemp = np.expand_dims(net[0, 0, :],axis=1) - t
print(sstemp)





print("----sss---------")
print("----fdsafdnksa---------")

print(net.shape)
print(t.shape)

temp = np.subtract(net,np.squeeze(t)) ** 2
temp = temp.sum(axis=2)
ee = np.where(temp == temp.min()) 
print(ee[0])
print(ee[1])
print(net[ee[0],ee[1]])

print("----sss---------")


print(t.shape)
print(net.shape)
temp = np.sqrt(net.dot(t) ** 2)
print(temp.shape)
ee = np.where(temp == temp.min()) 
print("-------------")
print(ee[0])
print(ee[1])
print(net[ee[0],ee[1]])



print("-------------")
sys.exit()





# -- end code ---