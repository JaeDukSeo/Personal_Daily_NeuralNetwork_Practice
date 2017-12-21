import numpy as np,sys

def sigmoid(x):
    return 1/ ( 1+ np.exp(-(x)))

def d_sigmoid(x):
    return sigmoid(x) * ( 1 - sigmoid(x))

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

w1 = 2*np.random.random((3,4)) - 1
w2 = 2*np.random.random((4,1)) - 1

og_w1 = w1
og_w2 = w2

for j in xrange(60000):

    layer_1 = X.dot(w1)
    layer_1_act = sigmoid(layer_1)

    final = layer_1_act.dot(w2)
    final_act = sigmoid(final)

    l2_error = (y - final_act)
    # if (j% 10000) == 0:
        # print "Error:" + str(np.mean(np.abs(l2_error)))

    grad_2_part_1 = (y - final_act)
    grad_2_part_2 = d_sigmoid(final)
    grad_2_part_3 = layer_1_act
    grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_sigmoid(layer_1)
    grad_1_part_3 = X
    grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    w2 += grad_2
    w1 += grad_1


layer_1 = X.dot(w1)
layer_1_act = sigmoid(layer_1)

final = layer_1_act.dot(w2)
final_act = sigmoid(final)
print  final_act,'\n---------'
print  w2,'\n---------'
print  w1,'\n---------'
print '-----------111111111------------'



    
# randomly initialize our weights with mean 0
syn0_2 = og_w1
syn1_2 = og_w2


for j in xrange(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0_2))
    l2 = sigmoid(np.dot(l1,syn1_2))

    # how much did we miss the target value?
    l2_error = y - l2
    
    # if (j% 10000) == 0:
        # print "Error:" + str(np.mean(np.abs(l2_error)))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much. d_sigmoid(l2)
    l2_delta = l2_error*d_sigmoid(l2)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1_2.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * d_sigmoid(l1)

    syn1_2 += l1.T.dot(l2_delta)
    syn0_2 += l0.T.dot(l1_delta)

l0 = X
l1 = sigmoid(np.dot(l0,syn0_2))
l2 = sigmoid(np.dot(l1,syn1_2))
print  l2,'\n---------'
print  syn1_2,'\n---------'
print  syn0_2,'\n---------'
print '-----------2222222------------'

# ---------- COMPOARE -----------

    
# randomly initialize our weights with mean 0
syn0 = og_w1
syn1 = og_w2

for j in xrange(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    # if (j% 10000) == 0:
        # print "Error:" + str(np.mean(np.abs(l2_error)))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

l0 = X
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))
print  l2,'\n---------'
print  syn1,'\n---------'
print  syn0,'\n---------'



syn0_3 = og_w1
syn1_3 = og_w2

for j in xrange(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0_3))
    l2 = nonlin(np.dot(l1,syn1_3))

    # how much did we miss the target value?
    l2_error = y - l2
    
    # if (j% 10000) == 0:
        # print "Error:" + str(np.mean(np.abs(l2_error)))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1_3 += l1.T.dot(l2_delta)
    syn0_3 += l0.T.dot(l1_delta)

l0 = X
l1 = nonlin(np.dot(l0,syn0_3))
l2 = nonlin(np.dot(l1,syn1_3))
print  l2,'\n---------'
print  syn1_3,'\n---------'
print  syn0_3,'\n---------'



# -------- END OF THE CODE ------