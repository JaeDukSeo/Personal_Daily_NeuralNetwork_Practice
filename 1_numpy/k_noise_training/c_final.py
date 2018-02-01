import numpy as np,sys,time
from mnist import MNIST
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(452)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# -1. All of our activation functions
def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * (1 - log(x))

def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1/( 1 + x **2)

# 0. Load Data
mndata = MNIST('../../z_MINST_DATA')
images, labels = mndata.load_testing()
images,labels = np.array(images),np.array(labels)
only_zero_index,only_one_index = np.where(labels==0)[0],np.where(labels==1)[0]

only_zero_image,only_zero_label = images[[only_zero_index]],np.expand_dims(labels[[only_zero_index]],axis=1)
only_one_image,only_one_label   = images[[only_one_index]],np.expand_dims(labels[[only_one_index]],axis=1)

images = np.vstack((only_zero_image,only_one_image))
labels = np.vstack((only_zero_label,only_one_label))
images,label = shuffle(images,labels)

test_image_num = 20
testing_images, testing_lables =images[:test_image_num,:],label[:test_image_num,:]
training_images,training_lables =images[test_image_num:,:],label[test_image_num:,:]

# 1. Declare Hyper Parameters
learning_rate = 0.03
learning_rate2 = 0.0001
time_change = 2
num_epoch = 100
value  = 0.2

time_array,miss_classiification = [],[]
cost_array =[]

# 2. Declare all of the weights
w1 = np.random.randn(784,840)*value
w2 = np.random.randn(840,1024)*value
w3 = np.random.randn(1024,1)*value

b1 = np.random.randn(840) *value
b2 = np.random.randn(1024)*value
b3 = np.random.randn(1)*value

# a. Gumbel Distribution
w1g,w2g,w3g = w1,w2,w3
b1g,b2g,b3g = b1,b2,b3

# b. Gaussian Distribution - Normal
w1n,w2n,w3n = w1,w2,w3
b1n,b2n,b3n = b1,b2,b3

# c. “standard normal” distribution.
w1s,w2s,w3s = w1,w2,w3
b1s,b2s,b3s = b1,b2,b3

# d. Binomial Distribution
w1b,w2b,w3b = w1,w2,w3
b1b,b2b,b3b = b1,b2,b3

# e. Beta Distribution
w1Beta,w2Beta,w3Beta = w1,w2,w3
b1Beta,b2Beta,b3Beta = b1,b2,b3

# f. poisson Distribution
w1p,w2p,w3p = w1,w2,w3
b1p,b2p,b3p = b1,b2,b3

# g. zipf Distribution
w1z,w2z,w3z = w1,w2,w3
b1z,b2z,b3z = b1,b2,b3

# h. pareto Distribution
w1pareto,w2pareto,w3pareto = w1,w2,w3
b1pareto,b2pareto,b3pareto = b1,b2,b3

# i. power Distribution
w1power,w2power,w3power = w1,w2,w3
b1power,b2power,b3power = b1,b2,b3

# j. rayleigh Distribution
w1rayleigh,w2rayleigh,w3rayleigh = w1,w2,w3
b1rayleigh,b2rayleigh,b3rayleigh = b1,b2,b3

# k. triangular Distribution
w1triangular,w2triangular,w3triangular = w1,w2,w3
b1triangular,b2triangular,b3triangular = b1,b2,b3

# l. weibull Distribution
w1weibull,w2weibull,w3weibull = w1,w2,w3
b1weibull,b2weibull,b3weibull = b1,b2,b3

# m. noncentral_chisquare Distribution
w1nc,w2nc,w3nc = w1,w2,w3
b1nc,b2nc,b3nc = b1,b2,b3

# n. Our Favourite Back Prop
w1backprop,w2backprop,w3backprop = w1,w2,w3
b1backprop,b2backprop,b3backprop = b1,b2,b3


# a. Gumbel Distribution Reset Learning Rate
learning_rate = 0.03
start = time.time()
cost_temp_array = []
print("Trainined Started for a. Gumbel Distribution ")
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1g) + b1g
    l1A = arctan(l1)

    l2 = l1A.dot(w2g) + b2g
    l2A = arctan(l2)

    l3 = l2A.dot(w3g) + b3g
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')

    gradient_weight_3 = np.random.gumbel(size=w3.shape)
    gradient_bias_3 = np.random.gumbel(size=b3.shape)

    gradient_weight_2 = np.random.gumbel(size=w2.shape)
    gradient_bias_2 = np.random.gumbel(size=b2.shape)

    gradient_weight_1 = np.random.gumbel(size=w1.shape)
    gradient_bias_1 = np.random.gumbel(size=b1.shape)
    
    w3g = w3g + 0.01 * learning_rate* cost * gradient_weight_3
    b3g = b3g + 0.01 *learning_rate* cost * gradient_bias_3

    w2g = w2g + 0.1 *learning_rate* cost * gradient_weight_2
    b2g = b2g + 0.1 * learning_rate*cost * gradient_bias_2

    w1g = w1g + 1.0 * learning_rate*cost * gradient_weight_1
    b1g = b1g + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for a. Gumbel Distribution")
l1 = testing_images.dot(w1g) + b1g
l1A = arctan(l1)

l2 = l1A.dot(w2g)+ b2g
l2A = arctan(l2)

l3 = l2A.dot(w3g)+ b3g
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label: ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# b. Gaussian Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for b. Gaussian Distribution")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1n) + b1n
    l1A = arctan(l1)

    l2 = l1A.dot(w2n) + b2n
    l2A = arctan(l2)

    l3 = l2A.dot(w3n) + b3n
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    cost_temp_array.append(cost)

    gradient_weight_3 = np.random.normal(size=w3.shape)
    gradient_bias_3 = np.random.normal(size=b3.shape)

    gradient_weight_2 = np.random.normal(size=w2.shape)
    gradient_bias_2 = np.random.normal(size=b2.shape)

    gradient_weight_1 = np.random.normal(size=w1.shape)
    gradient_bias_1 = np.random.normal(size=b1.shape)
    
    w3n = w3n + 0.01 * learning_rate* cost * gradient_weight_3
    b3n = b3n + 0.01 *learning_rate* cost * gradient_bias_3

    w2n = w2n + 0.1 *learning_rate* cost * gradient_weight_2
    b2n = b2n + 0.1 * learning_rate*cost * gradient_bias_2

    w1n = w1n + 1.0 * learning_rate*cost * gradient_weight_1
    b1n = b1n + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for b. Gaussian Distribution")
l1 = testing_images.dot(w1n) + b1n
l1A = arctan(l1)

l2 = l1A.dot(w2n)+ b2n
l2A = arctan(l2)

l3 = l2A.dot(w3n)+ b3n
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label: ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# c. “standard normal” distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for c. “standard normal” distribution ")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1s) + b1s
    l1A = arctan(l1)

    l2 = l1A.dot(w2s) + b2s
    l2A = arctan(l2)

    l3 = l2A.dot(w3s) + b3s
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    cost_temp_array.append(cost)

    gradient_weight_3 = np.random.randn(w3.shape[0],w3.shape[1])
    gradient_bias_3 = np.random.randn(b3.shape[0])

    gradient_weight_2 = np.random.randn(w2.shape[0],w2.shape[1])
    gradient_bias_2 = np.random.randn(b2.shape[0])

    gradient_weight_1 = np.random.randn(w1.shape[0],w1.shape[1])
    gradient_bias_1 = np.random.randn(b1.shape[0])
    
    w3s = w3s + 0.01 * learning_rate* cost * gradient_weight_3
    b3s = b3s + 0.01 *learning_rate* cost * gradient_bias_3

    w2s = w2s + 0.1 *learning_rate* cost * gradient_weight_2
    b2s = b2s + 0.1 * learning_rate*cost * gradient_bias_2

    w1s = w1s + 1.0 * learning_rate*cost * gradient_weight_1
    b1s = b1s + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for c. “standard normal” distribution ")
l1 = testing_images.dot(w1s) + b1s
l1A = arctan(l1)

l2 = l1A.dot(w2s)+ b2s
l2A = arctan(l2)

l3 = l2A.dot(w3s)+ b3s
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label: ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# d. Binomial Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for d. Binomial Distribution")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1b) + b1b
    l1A = arctan(l1)

    l2 = l1A.dot(w2b) + b2b
    l2A = arctan(l2)

    l3 = l2A.dot(w3n) + b3b
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    cost_temp_array.append(cost)
    
    n, p = 1, .5 
    gradient_weight_3 = np.random.binomial(n, p,size=w3.shape)
    gradient_bias_3 = np.random.binomial(n, p,size=b3.shape)

    gradient_weight_2 = np.random.binomial(n, p,size=w2.shape)
    gradient_bias_2 = np.random.binomial(n, p,size=b2.shape)

    gradient_weight_1 = np.random.binomial(n, p,size=w1.shape)
    gradient_bias_1 = np.random.binomial(n, p,size=b1.shape)
    
    w3b = w3b + 0.01 * learning_rate* cost * gradient_weight_3
    b3b = b3b + 0.01 *learning_rate* cost * gradient_bias_3

    w2b = w2b + 0.1 *learning_rate* cost * gradient_weight_2
    b2b = b2b + 0.1 * learning_rate*cost * gradient_bias_2

    w1b = w1b + 1.0 * learning_rate*cost * gradient_weight_1
    b1b = b1b + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for d. Binomial Distribution")
l1 = testing_images.dot(w1b) + b1b
l1A = arctan(l1)

l2 = l1A.dot(w2b)+ b2b
l2A = arctan(l2)

l3 = l2A.dot(w3b)+ b3b
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# e. Beta Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for e. Beta Distribution")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1Beta) + b1Beta
    l1A = arctan(l1)

    l2 = l1A.dot(w2Beta) + b2Beta
    l2A = arctan(l2)

    l3 = l2A.dot(w3Beta) + b3Beta
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    n, p = 1, .5 
    gradient_weight_3 = np.random.beta(n, p,size=w3.shape)
    gradient_bias_3 = np.random.beta(n, p,size=b3.shape)

    gradient_weight_2 = np.random.beta(n, p,size=w2.shape)
    gradient_bias_2 = np.random.beta(n, p,size=b2.shape)

    gradient_weight_1 = np.random.beta(n, p,size=w1.shape)
    gradient_bias_1 = np.random.beta(n, p,size=b1.shape)
    
    w3Beta = w3Beta + 0.01 * learning_rate* cost * gradient_weight_3
    b3Beta = b3Beta + 0.01 *learning_rate* cost * gradient_bias_3

    w2Beta = w2Beta + 0.1 *learning_rate* cost * gradient_weight_2
    b2Beta = b2Beta + 0.1 * learning_rate*cost * gradient_bias_2

    w1Beta = w1Beta + 1.0 * learning_rate*cost * gradient_weight_1
    b1Beta = b1Beta + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for e. Beta Distribution")
l1 = testing_images.dot(w1Beta) + b1Beta
l1A = arctan(l1)

l2 = l1A.dot(w2Beta)+ b2Beta
l2A = arctan(l2)

l3 = l2A.dot(w3Beta)+ b3Beta
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# f. poisson Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for f. poisson Distribution")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1p) + b1p
    l1A = arctan(l1)

    l2 = l1A.dot(w2p) + b2p
    l2A = arctan(l2)

    l3 = l2A.dot(w3p) + b3p
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    cost_temp_array.append(cost)
    n, p = 1, .5 
    gradient_weight_3 = np.random.poisson(n, size=w3.shape)
    gradient_bias_3 = np.random.poisson(n, size=b3.shape)

    gradient_weight_2 = np.random.poisson(n, size=w2.shape)
    gradient_bias_2 = np.random.poisson(n, size=b2.shape)

    gradient_weight_1 = np.random.poisson(n, size=w1.shape)
    gradient_bias_1 = np.random.poisson(n, size=b1.shape)
    
    w3p = w3p + 0.01 * learning_rate* cost * gradient_weight_3
    b3p = b3p + 0.01 *learning_rate* cost * gradient_bias_3

    w2p = w2p + 0.1 *learning_rate* cost * gradient_weight_2
    b2p = b2p + 0.1 * learning_rate*cost * gradient_bias_2

    w1p = w1p + 1.0 * learning_rate*cost * gradient_weight_1
    b1p = b1p + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for f. poisson Distribution")
l1 = testing_images.dot(w1p) + b1p
l1A = arctan(l1)

l2 = l1A.dot(w2p)+ b2p
l2A = arctan(l2)

l3 = l2A.dot(w3p)+ b3p
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# g. zipf Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for g. zipf Distribution")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1z) + b1z
    l1A = arctan(l1)

    l2 = l1A.dot(w2z) + b2z
    l2A = arctan(l2)

    l3 = l2A.dot(w3z) + b3z
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    n, p = 2., .5 
    gradient_weight_3 = np.random.zipf(n, size=w3.shape)
    gradient_bias_3 = np.random.zipf(n, size=b3.shape)

    gradient_weight_2 = np.random.zipf(n, size=w2.shape)
    gradient_bias_2 = np.random.zipf(n, size=b2.shape)

    gradient_weight_1 = np.random.zipf(n, size=w1.shape)
    gradient_bias_1 = np.random.zipf(n, size=b1.shape)
    
    w3z = w3z + 0.01 * learning_rate* cost * gradient_weight_3
    b3z = b3z + 0.01 *learning_rate* cost * gradient_bias_3

    w2z = w2z + 0.1 *learning_rate* cost * gradient_weight_2
    b2z = b2z + 0.1 * learning_rate*cost * gradient_bias_2

    w1z = w1z + 1.0 * learning_rate*cost * gradient_weight_1
    b1z = b1z + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for g. zipf Distribution")
l1 = testing_images.dot(w1z) + b1z
l1A = arctan(l1)

l2 = l1A.dot(w2z)+ b2z
l2A = arctan(l2)

l3 = l2A.dot(w3z)+ b3z
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# h. pareto Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for h. pareto Distribution")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1pareto) + b1pareto
    l1A = arctan(l1)

    l2 = l1A.dot(w2pareto) + b2pareto
    l2A = arctan(l2)

    l3 = l2A.dot(w3pareto) + b3pareto
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    n, p = 1, .5 
    gradient_weight_3 = np.random.pareto(n, size=w3.shape)
    gradient_bias_3 = np.random.pareto(n, size=b3.shape)

    gradient_weight_2 = np.random.pareto(n, size=w2.shape)
    gradient_bias_2 = np.random.pareto(n, size=b2.shape)

    gradient_weight_1 = np.random.pareto(n, size=w1.shape)
    gradient_bias_1 = np.random.pareto(n, size=b1.shape)
    
    w3pareto = w3pareto + 0.01 * learning_rate* cost * gradient_weight_3
    b3pareto = b3pareto + 0.01 *learning_rate* cost * gradient_bias_3

    w2pareto = w2pareto + 0.1 *learning_rate* cost * gradient_weight_2
    b2pareto = b2pareto + 0.1 * learning_rate*cost * gradient_bias_2

    w1pareto = w1pareto + 1.0 * learning_rate*cost * gradient_weight_1
    b1pareto = b1pareto + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for h. pareto Distribution")
l1 = testing_images.dot(w1pareto) + b1pareto
l1A = arctan(l1)

l2 = l1A.dot(w2pareto)+ b2pareto
l2A = arctan(l2)

l3 = l2A.dot(w3pareto)+ b3pareto
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# i. power Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for i. power Distribution ")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1power) + b1power
    l1A = arctan(l1)

    l2 = l1A.dot(w2power) + b2power
    l2A = arctan(l2)

    l3 = l2A.dot(w3power) + b3power
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    n, p = 1, .5 
    gradient_weight_3 = np.random.power(n, size=w3.shape)
    gradient_bias_3 = np.random.power(n, size=b3.shape)

    gradient_weight_2 = np.random.power(n, size=w2.shape)
    gradient_bias_2 = np.random.power(n, size=b2.shape)

    gradient_weight_1 = np.random.power(n, size=w1.shape)
    gradient_bias_1 = np.random.power(n, size=b1.shape)
    
    w3power = w3power + 0.01 * learning_rate* cost * gradient_weight_3
    b3power = b3power + 0.01 *learning_rate* cost * gradient_bias_3

    w2power = w2power + 0.1 *learning_rate* cost * gradient_weight_2
    b2power = b2power + 0.1 * learning_rate*cost * gradient_bias_2

    w1power = w1power + 1.0 * learning_rate*cost * gradient_weight_1
    b1power = b1power + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for # i. power Distribution ")
l1 = testing_images.dot(w1power) + b1power
l1A = arctan(l1)

l2 = l1A.dot(w2power)+ b2power
l2A = arctan(l2)

l3 = l2A.dot(w3power)+ b3power
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

#  j. rayleigh Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for j. rayleigh Distribution ")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1rayleigh) + b1rayleigh
    l1A = arctan(l1)

    l2 = l1A.dot(w2rayleigh) + b2rayleigh
    l2A = arctan(l2)

    l3 = l2A.dot(w3rayleigh) + b3rayleigh
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    n, p = 1, .5 
    gradient_weight_3 = np.random.rayleigh(n, size=w3.shape)
    gradient_bias_3 = np.random.rayleigh(n, size=b3.shape)

    gradient_weight_2 = np.random.rayleigh(n, size=w2.shape)
    gradient_bias_2 = np.random.rayleigh(n, size=b2.shape)

    gradient_weight_1 = np.random.rayleigh(n, size=w1.shape)
    gradient_bias_1 = np.random.rayleigh(n, size=b1.shape)
    
    w3rayleigh = w3rayleigh + 0.01 * learning_rate* cost * gradient_weight_3
    b3rayleigh = b3rayleigh + 0.01 *learning_rate* cost * gradient_bias_3

    w2rayleigh = w2rayleigh + 0.1 *learning_rate* cost * gradient_weight_2
    b2rayleigh = b2rayleigh + 0.1 * learning_rate*cost * gradient_bias_2

    w1rayleigh = w1rayleigh + 1.0 * learning_rate*cost * gradient_weight_1
    b1rayleigh = b1rayleigh + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for j. rayleigh Distribution ")
l1 = testing_images.dot(w1rayleigh) + b1rayleigh
l1A = arctan(l1)

l2 = l1A.dot(w2rayleigh)+ b2rayleigh
l2A = arctan(l2)

l3 = l2A.dot(w3rayleigh)+ b3rayleigh
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

#  k. triangular Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for k. triangular Distribution ")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1triangular) + b1triangular
    l1A = arctan(l1)

    l2 = l1A.dot(w2triangular) + b2triangular
    l2A = arctan(l2)

    l3 = l2A.dot(w3triangular) + b3triangular
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    n, p = 1, .5 
    gradient_weight_3 = np.random.triangular(-1,0,1, size=w3.shape)
    gradient_bias_3 = np.random.triangular(-1,0,1, size=b3.shape)

    gradient_weight_2 = np.random.triangular(-1,0,1, size=w2.shape)
    gradient_bias_2 = np.random.triangular(-1,0,1, size=b2.shape)

    gradient_weight_1 = np.random.triangular(-1,0,1, size=w1.shape)
    gradient_bias_1 = np.random.triangular(-1,0,1, size=b1.shape)
    
    w3triangular = w3triangular + 0.01 * learning_rate* cost * gradient_weight_3
    b3triangular = b3triangular + 0.01 *learning_rate* cost * gradient_bias_3

    w2triangular = w2triangular + 0.1 *learning_rate* cost * gradient_weight_2
    b2triangular = b2triangular + 0.1 * learning_rate*cost * gradient_bias_2

    w1triangular = w1triangular + 1.0 * learning_rate*cost * gradient_weight_1
    b1triangular = b1triangular + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for k. triangular Distribution ")
l1 = testing_images.dot(w1triangular) + b1triangular
l1A = arctan(l1)

l2 = l1A.dot(w2triangular)+ b2triangular
l2A = arctan(l2)

l3 = l2A.dot(w3triangular)+ b3triangular
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# l. weibull Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for l. weibull Distribution ")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1weibull) + b1weibull
    l1A = arctan(l1)

    l2 = l1A.dot(w2weibull) + b2weibull
    l2A = arctan(l2)

    l3 = l2A.dot(w3weibull) + b3weibull
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    n, p = 1, .5 
    gradient_weight_3 = np.random.weibull(n, size=w3.shape)
    gradient_bias_3 = np.random.weibull(n, size=b3.shape)

    gradient_weight_2 = np.random.weibull(n, size=w2.shape)
    gradient_bias_2 = np.random.weibull(n, size=b2.shape)

    gradient_weight_1 = np.random.weibull(n, size=w1.shape)
    gradient_bias_1 = np.random.weibull(n, size=b1.shape)

    w3weibull = w3weibull + 0.01 * learning_rate* cost * gradient_weight_3
    b3weibull = b3weibull + 0.01 *learning_rate* cost * gradient_bias_3

    w2weibull = w2weibull + 0.1 *learning_rate* cost * gradient_weight_2
    b2weibull = b2weibull + 0.1 * learning_rate*cost * gradient_bias_2

    w1weibull = w1weibull + 1.0 * learning_rate*cost * gradient_weight_1
    b1weibull = b1weibull + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for l. weibull Distribution")
l1 = testing_images.dot(w1weibull) + b1weibull
l1A = arctan(l1)

l2 = l1A.dot(w2weibull)+ b2weibull
l2A = arctan(l2)

l3 = l2A.dot(w3weibull)+ b3weibull
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

# m. noncentral_chisquare Distribution - Reset Learning Rate
learning_rate = 0.03
print("Trainined Started for m. noncentral_chisquare Distribution ")
start = time.time()
cost_temp_array = []
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1nc) + b1nc
    l1A = arctan(l1)

    l2 = l1A.dot(w2nc) + b2nc
    l2A = arctan(l2)

    l3 = l2A.dot(w3nc) + b3nc
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')
    n, p = 1, .5 
    gradient_weight_3 = np.random.noncentral_chisquare(n,p, size=w3.shape)
    gradient_bias_3 = np.random.noncentral_chisquare(n,p, size=b3.shape)

    gradient_weight_2 = np.random.noncentral_chisquare(n,p, size=w2.shape)
    gradient_bias_2 = np.random.noncentral_chisquare(n, p,size=b2.shape)

    gradient_weight_1 = np.random.noncentral_chisquare(n, p,size=w1.shape)
    gradient_bias_1 = np.random.noncentral_chisquare(n, p,size=b1.shape)

    w3nc = w3nc + 0.01 * learning_rate* cost * gradient_weight_3
    b3nc = b3nc + 0.01 *learning_rate* cost * gradient_bias_3

    w2nc = w2nc + 0.1 *learning_rate* cost * gradient_weight_2
    b2nc = b2nc + 0.1 * learning_rate*cost * gradient_bias_2

    w1nc = w1nc + 1.0 * learning_rate*cost * gradient_weight_1
    b1nc = b1nc + 1.0 * learning_rate*cost * gradient_bias_1
    
    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for m. noncentral_chisquare Distribution")
l1 = testing_images.dot(w1nc) + b1nc
l1A = arctan(l1)

l2 = l1A.dot(w2nc)+ b2nc
l2A = arctan(l2)

l3 = l2A.dot(w3nc)+ b3nc
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')


# n. Back Prop - Reset Learning Rate
learning_rate = 0.03
start = time.time()
cost_temp_array = []
print("Trainined Started for n. Back Prop ")
for iter in range(num_epoch):
    
    l1 = training_images.dot(w1backprop) + b1backprop
    l1A = arctan(l1)

    l2 = l1A.dot(w2backprop) + b2backprop
    l2A = arctan(l2)

    l3 = l2A.dot(w3backprop) + b3backprop
    l3A = log(l3)

    cost = np.square(l3A - training_lables).sum() / len(training_images)
    cost_temp_array.append(cost)
    print("Current Iter: ",iter, " Current cost :", cost,end='\r')

    grad_3_part_1 = l3A - training_lables
    grad_3_part_2 = d_log(l3)
    grad_3_part_3 = l2A
    grad_3_w = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)
    grad_3_b = np.ones((2095,1)).T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3backprop.T)
    grad_2_part_2 = d_arctan(l2)
    grad_2_part_3 = l1A
    grad_2_w = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
    grad_2_b = np.ones((2095,1)).T.dot(grad_2_part_1 * grad_2_part_2)

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2backprop.T)
    grad_1_part_2 = d_arctan(l1)
    grad_1_part_3 = training_images
    grad_1_w = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)
    grad_1_b = np.ones((2095,1)).T.dot(grad_1_part_1 * grad_1_part_2)

    w3backprop = w3backprop - learning_rate * grad_3_w
    b3backprop = b3backprop - learning_rate * grad_3_b

    w2backprop = w2backprop - learning_rate * grad_2_w
    b2backprop = b2backprop - learning_rate * grad_2_b

    w1backprop = w1backprop - learning_rate * grad_1_w
    b1backprop = b1backprop - learning_rate * grad_1_b

    if iter > time_change:
        learning_rate = learning_rate2
end = time.time()

print('\n')
print("Results for n. Back Prop")
l1 = testing_images.dot(w1backprop) + b1backprop
l1A = arctan(l1)

l2 = l1A.dot(w2backprop) + b2backprop
l2A = arctan(l2)

l3 = l2A.dot(w3backprop) + b3backprop
l3A = log(l3)
print("Ground Truth Label: ",testing_lables.T)
print("Predicted Label:    ",np.round(l3A).astype(int).T)
print("Training Time: ", end - start)
time_array.append(end - start)
miss_classiification.append(test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum() )
cost_array.append(cost_temp_array)
print("# of  Misclassified Images: " ,test_image_num - (testing_lables.T ==np.round(l3A).astype(int).T ).sum()  )
print(l3A.T)
print('\n')

bar_color = ['b', 'g', 'saddlebrown', 'steelblue', 
            'orangered', 'y', 'paleturquoise', 'royalblue',
            'salmon','silver','skyblue','slateblue','peru','plum']

labels_z = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']

for i in range(len(cost_array)):
    plt.plot(np.arange(num_epoch), cost_array[i],color=bar_color[i],linewidth=4,label=labels_z[i])
plt.title("Cost per Training")
plt.savefig("Cost_per_Training.png")
plt.close('all')

plt.bar(np.arange(14), time_array,color=bar_color)
plt.xticks(np.arange(14), ('a','b','c','d','e','f','g','h','i','j','k','l','m','n'))
plt.title("Training Time Bar Graph in Seconds")
plt.savefig("Training_Time_Bar.png")
plt.close('all')

plt.bar(np.arange(14), miss_classiification,color=bar_color)
plt.xticks(np.arange(14), ('a','b','c','d','e','f','g','h','i','j','k','l','m','n'))
plt.title("Misclassification Bar Graph")
plt.savefig("Misclassification_Bar_Graph.png")
plt.close('all')

print('a','b','c','d','e','f','g','h','i','j','k','l','m','n')
print(time_array)
print(miss_classiification)

















# -- end code --