import numpy as np,sys
import matplotlib.pyplot as plt
from matplotlib import patches as patches
np.random.seed(8)
np.set_printoptions(precision=2)

raw_data = np.random.randint(0, 255, (100, 3))
network_dimensions = np.array([5, 5])
n_iterations = 2000
init_learning_rate = 0.1
# establish size variables based on data
m = raw_data.shape[0]
n = raw_data.shape[1]

# weight matrix (i.e. the SOM) needs to be one m-dimensional vector for each neuron in the SOM
net = np.random.random((n,network_dimensions[0], network_dimensions[1]))
# initial neighbourhood radius
init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
# radius decay parameter
time_constant = n_iterations / np.log(init_radius)
normalise_data = True
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



for i in range(n_iterations):

    current_data = data[np.random.randint(0,len(data)),:]

    temp = np.subtract(net.T,current_data).T ** 2
    temp_sum = temp.sum(axis=0)
    bmu_idx = np.squeeze(np.asarray(np.where(temp_sum == temp_sum.min())))

    r = init_radius * np.exp(-i / time_constant)
    l = init_learning_rate * np.exp(-i / n_iterations)
    effect_matrix = np.zeros((5,5))

    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
            if w_dist <= r**2:
                effect_matrix[x,y] = np.exp(-w_dist / (2* (r**2)))

    effect_matrix = np.repeat(np.expand_dims(effect_matrix.T,axis=2), 3,axis=2).T
    net = net - (l * (effect_matrix.T * current_data).T  * net)



print(net.shape)

fig = plt.figure()
# setup axes
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim((0, net.shape[0]+1))
ax.set_ylim((0, net.shape[1]+1))
ax.set_title('Self-Organising Map after %d iterations' % n_iterations)

# plot the rectangles
for x in range(1, net.shape[1] + 1):
    for y in range(1, net.shape[2] + 1):
        ax.add_patch(patches.Rectangle((x-1, y-0.5), 1, 1,facecolor=net[:,x-1,y-1],edgecolor='none'))
plt.show()













# sys.exit()
# # ---- Training Aspect -----
# for i in range(n_iterations):
#     # select a training example at random
#     t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))
#     temp = np.subtract(np.squeeze(t).T,net[:,0,0]) ** 2
#     temp_sum = temp.sum(axis=2)
#     bmu_idx = np.squeeze(np.asarray(np.where(temp_sum == temp_sum.min())))
#     bmu = net[bmu_idx[0],bmu_idx[1],:].reshape(np.array([m, 1]))

#     # decay the SOM parameters
#     r = init_radius * np.exp(-i / time_constant)
#     l = init_learning_rate * np.exp(-i / n_iterations)

#     # Create Effect Matrix
#     for x in range(net.shape[0]):
#         for y in range(net.shape[1]):
#             w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
#             if w_dist <= r**2:
#                 effect_matrix[x,y] = np.exp(-w_dist / (2* (r**2)))
#     print(effect_matrix.shape)
            
#     effect_matrix = np.repeat(np.expand_dims(effect_matrix,axis=2), 3,axis=2)
#     print(net)
    

#     sys.exit()


#     # now we know the BMU, update its weight vector to move closer to input
#     # and move its neighbours in 2-D space closer
#     # by a factor proportional to their 2-D distance from the BMU
#     for x in range(net.shape[0]):
#         for y in range(net.shape[1]):
#             w = net[x, y, :].reshape(m, 1)
#             # get the 2-D distance (again, not the actual Euclidean distance)
#             w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
#             # if the distance is within the current neighbourhood radius
#             if w_dist <= r**2:
#                 # calculate the degree of influence (based on the 2-D distance)
#                 influence = np.exp(-w_dist / (2* (r**2)))

#                 # now update the neuron's weight using the formula:
#                 # new w = old w + (learning rate * influence * delta)
#                 # where delta = input vector (t) - old w
#                 new_w = w + (l * influence * (t - w))
#                 # commit the new weight
#                 net[x, y, :] = new_w.reshape(1, 3)

































# --------------- trash -------


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