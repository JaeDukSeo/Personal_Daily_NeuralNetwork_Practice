import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(  (-1) * x)) 

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,0,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(1):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1

    grad_1_part_1 = l1_error
    grad_1_part_2 = d_sigmoid(np.dot(l0,syn0))
    grad_1_part_3 = l0.T

    # print grad_1_part_1.shape
    # print grad_1_part_2.shape
    # print grad_1_part_3.shape
    
    print l1_error.shape
    print nonlin(l1,True).shape
    print (l1_error * nonlin(l1,True)).shape
    print '------HAVE TO KNOW!---'
    
    l1_delta = l1_error * grad_1_part_2
    print l1_delta.shape
    print l0.T.shape

    # update weights
    syn0 += np.dot(l0.T,l1_delta)
    print 'TARGET : -----',syn0.shape


    # forward propagation
    l0 = np.dot(X,syn0)
    l0_sigmoid  = sigmoid(l0)
    l1 = l0

    # how much did we miss?
    l1_error = np.square(y - l1).sum()

    error = 2.0 * (y - l1)

    grad_1_part_1 = error
    grad_1_part_2 = d_sigmoid(l0)
    grad_1_part_3 = X

    print grad_1_part_1.shape
    print grad_1_part_2.shape
    print grad_1_part_3.shape
    print 'Target : ',syn0.shape
    

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights

    syn0 = syn0 + np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1
