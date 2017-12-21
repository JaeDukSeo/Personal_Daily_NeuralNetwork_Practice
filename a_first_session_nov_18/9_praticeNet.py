import numpy as np,sys

np.random.seed(3)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
number_of_epoch = 1000


for j in xrange(60000):

    l1 = X.dot(syn0)
    l1_act = sigmoid(l1)

    l2 = l1_act.dot(syn1)
    l2_act = sigmoid(l2)

    grad_2_part_1  = (y - l2_act)
    grad_2_part_2  = d_sigmoid(l2)
    grad_2_part_3  = l1_act
    grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

    grad_1_part_1  = (grad_2_part_1 * grad_2_part_2).dot(syn1.T)
    grad_1_part_2  = d_sigmoid(l1)
    grad_1_part_3  = X
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    syn1 += grad_2
    syn0 += grad_1

l1 = X.dot(syn0)
l1_act = sigmoid(l1)

l2 = l1_act.dot(syn1)
l2_act = sigmoid(l2)
print l2_act


# ----------- BELOW IS THE MOID CODE ---------
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):
    
    l1_temp = X.dot(syn0)
    l1 = sigmoid(l1_temp)
    # l1 = 1/(1+np.exp(-(l1_temp)))
    
    l2_temp = l1.dot(syn1)
    l2 = sigmoid(l2_temp)
    # l2 = 1/(1+np.exp(-(l2_temp)))
    
    grad_2_part_1 = (y - l2)
    grad_2_part_2 = d_sigmoid(l2_temp)
    grad_2_part_3 = l1
    grad_2 = grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

    l2_delta = grad_2_part_1*grad_2_part_2
    

    grad_1_part_1 = (grad_2_part_1*grad_2_part_2).dot(syn1.T)
    grad_1_part_2 = d_sigmoid(l1_temp)
    grad_1_part_3 = X
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    l1_delta = l2_delta.dot(syn1.T) * grad_1_part_2
    

    syn1 += grad_2
    syn0 += grad_1

l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
print '--- mod 1-----'
print l2



# ----------- BELOW IS THE ORIGINAL CODE ---------
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):
    
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))

    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
print '--- OG 1-----'
print l2






# ------------------------------------