import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x) ) 

def d_sigmoid(x):
    return sigmoid(x) * (1- sigmoid(x))

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T

input_d, h1_d,out_d = 3,4,1

og_w1 = 2*np.random.random((3,h1_d)) - 1
og_w2 = 2*np.random.random((h1_d,1)) - 1

w1 = og_w1
w2 = og_w2


learning_rate = 0.5


for j in xrange(60000):
    
    layer_1 = X.dot(w1)
    layer_1_act = sigmoid(layer_1)

    final = layer_1_act.dot(w2)
    final_act = sigmoid(final)

    grad_2_part_1 = (final_act - y)
    grad_2_part_2 = d_sigmoid(final)
    grad_2_part_3 = layer_1_act
    grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
    
    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = X
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    w2 -= learning_rate * grad_2
    w1 -= learning_rate * grad_1

print w2
print w1
print final_act
print '-----'
    
alpha,hidden_dim = (0.5,4)
synapse_0 = og_w1
synapse_1 = og_w2
for j in xrange(60000):
    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
    layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
    synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
    synapse_0 -= (alpha * X.T.dot(layer_1_delta))


print synapse_1
print synapse_0
print layer_2
print '-----'




# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
    
# input dataset
X = np.array([  [0,1],
                [0,1],
                [1,0],
                [1,0] ])
    
# output dataset            
y = np.array([[0,0,1,0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
synapse_0 = 2*np.random.random((2,1)) - 1

for iter in xrange(10000):

    # forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0,synapse_0))

    # how much did we miss?
    layer_1_error = layer_1 - y

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    synapse_0_derivative = np.dot(layer_0.T,layer_1_delta)

    # update weights
    synapse_0 -= synapse_0_derivative

print "Output After Training:"
print layer_1






# -------- END OF THE CODE ----