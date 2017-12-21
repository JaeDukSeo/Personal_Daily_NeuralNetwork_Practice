import numpy as np

def sigmoid(x):
    return 1/ ( 1+ np.exp(-(x)))

def d_sigmoid(x):
    return sigmoid(x) * ( 1 - sigmoid(x))

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[1,0,0,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(4)


# initialize weights randomly with mean 0
w1 = 2*np.random.random((3,1)) - 1


for iter in xrange(10000):
    layer_1 = X.dot(w1)
    layer_1_act = sigmoid(layer_1)

    grad_1_part_1 = (y - layer_1_act)
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = X
    grad_1  = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)
    
    w1 += grad_1

print "Output After Training:"
print layer_1_act,"\n-----------"
print w1


print '-------------- COMPARE ----------'
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    


# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1,"\n-----------"
print w1





# --------- END OF THE CODE er