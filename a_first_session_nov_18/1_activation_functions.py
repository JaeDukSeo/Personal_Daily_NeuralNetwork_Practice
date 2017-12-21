import numpy as np,sys

#------ OUR ACTIVATION FUNCTIONS and it derivatives ------
def sigmoid(x):
    return 1 / (1 + np.exp(  (-1) * x)) 

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

#  Func: Very simple relu layer function
def relu(x):
    temp = x>0
    return x * temp

    # if x > 0:
    #     return x
    # else: 
    #     return 0

def d_relu(x):
    temp = x>0
    return temp

    # if x > 0:
    #     return 1
    # else: 
    #     return 0

# Func: Leaky Relu
def leaky_relu(x):
    if x > 0:
        return 1 * x
    else: 
        return  0.01 * x
def d_leaky_relu(x):
    if x > 0:
        return 1 
    else: 
        return  0.01 

# Func: ELU 
def elu(x,alpha=1.0):
    if x > 0 :
        return x
    else:
        return alpha * (np.exp(x)-1)

def d_elu(x,alpha=1.0):
    if x > 0 :
        return 1
    else:
        # IT IS VERY IMPORTANT TO NOTE THAT the actual thing is alpha * np.exp(X)
        return alpha + elu(x,alpha)
#------ OUR ACTIVATION FUNCTIONS and it derivatives ------


# 1. Generate Training Data and Ground truth and operation and hyper parameter
np.random.seed(1)
input_d,hidden1_d,out_d = 3,3,1
number_of_epoch = 100
learning_rate = 1.0
x = np.array([
       [0,0,1],
       [0,1,1],
       [1,0,1],
       [1,1,1] 
])
y = np.array([[0],
             [1],
             [0],
             [1]])

# 2. Create weight
w1 = np.random.randn(input_d, hidden1_d) 
w2 = np.random.randn(hidden1_d, out_d) 

for i in range(number_of_epoch):

    for ii in range(10):
        layer_1 = x.dot(w1)
        layer_1_sig = sigmoid(layer_1)

        layer_2 = layer_1_sig.dot(w2)
        layer_2_sig = sigmoid(layer_2)

        error_cost = np.square(layer_2_sig - y)

        grad_2_part_1 = 2.0 * (layer_2_sig - y)
        grad_2_part_2 = d_sigmoid(layer_2)
        grad_2_part_3 = layer_1_sig
        grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3).T

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = x
        grad_1 = (grad_1_part_1 * grad_1_part_2).T.dot(grad_1_part_3)

        w1 = w1 - learning_rate * grad_1
        w2 = w2 - learning_rate * grad_2


layer_1 = x.dot(w1)
layer_1_sig = sigmoid(layer_1)
layer_2 = layer_1_sig.dot(w2)
layer_2_sig = sigmoid(layer_2)
print layer_2_sig > 0.5














sys.exit()
# ------------------------ RETAKE ON THE NN -----------------------
# 1. Generate Training Data and Ground truth and operation and hyper parameter
input_d,hidden1_d,out_d = 3,3,1
number_of_epoch = 1
learning_rate = 0.5
x = np.array([
       [0,0,1],
       [0,1,1],
       [1,0,1],
       [1,1,1] 
])
y = np.array([[0],
             [0],
             [0],
             [1]])

# 2. Create weight
w1 = np.random.randn(input_d, hidden1_d) 
w2 = np.random.randn(hidden1_d, out_d) 

# 3. Create Epoch
for i in range(number_of_epoch):
    
    for ii in range(1000):
        layer_1 = np.dot(x,w1)
        layer_1_act = sigmoid(layer_1)

        final_layer = np.dot(layer_1_act ,w2)
        final_layer_sigmoid = sigmoid(final_layer)

        # Cost function
        cost = np.square(final_layer_sigmoid - y).sum()

        # Back propagation
        error_cost = 2.0 * (final_layer - y)

        grad_2_part_1 = error_cost
        grad_2_part_2 = sigmoid(final_layer) * (1 - sigmoid(final_layer))
        grad_2_part_3 = layer_1_act
        grad_2 = (grad_2_part_1 * grad_2_part_2).T.dot(grad_2_part_3)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = x
        grad_1 = (grad_1_part_1.T * grad_1_part_2.T).dot(x)
        
        # Update Weight
        w1 = w1 - learning_rate * grad_1
        w2 = w2 - learning_rate * grad_2
        

    layer_1 = np.dot(x,w1)
    print layer_1.shape
    
    layer_1_act = sigmoid(layer_1)
    print layer_1_act.shape
    
    final_layer = np.dot(layer_1_act ,w2)
    print final_layer.shape
    
    final_layer_sigmoid = sigmoid(final_layer)
    print final_layer_sigmoid.shape

sys.exit()

# 0. Generate Numpy Data - and create hyperparameter
N, D_in, H, D_out = 64, 1000, 100, 10
x = np.random.randn(N, D_in) # [64,1000]
y = np.random.randn(N, D_out) # [64,10] - so we have 10 things..?
learning_rate = 1e-6
epoch  = 1000

# 1. Create the weights
# Randomly initialize weights - create random weigths
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

for i in range(epoch):
    # 2. Create the layers and opertions
    hidden1 = x.dot(w1)
    hidden1_relu = np.maximum(0,hidden1)
    final = hidden1_relu.dot(w2)

    # 3. Loop and cost functions
    cost = np.square(y-final).sum()
    print(i, cost)

    # 4. Perform Back propagation Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (final - y)
    grad_w2 = hidden1_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[hidden1 < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 = w1 -  learning_rate * grad_w1
    w2 = w2 -  learning_rate * grad_w2

print('------------Sigmoid- FAILED TO BACK PROPAGAT AT THE MOMENT ------------')
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

for i in range(epoch):
    # 2. Create the layers and opertions
    hidden1 = x.dot(w1)
    hidden1_sigmoid = sigmoid(hidden1)
    final = hidden1_sigmoid.dot(w2)

    # 3. Loop and cost functions
    cost = np.square(y-final).sum()
    print(i, cost)

    # 4. Perform Back propagation Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (final - y)
    grad_final_sigmoid = sigmoid(hidden1_sigmoid.dot(w2)) * (1 - sigmoid(hidden1_sigmoid.dot(w2)))
    grad_w2 = grad_y_pred.dot(grad_final_sigmoid.T).dot(hidden1_sigmoid)

    print grad_y_pred.shape
    print grad_final_sigmoid.shape
    print hidden1_sigmoid.shape
    print '--------'

    grad_first_sigmoid = sigmoid(x.dot(w1)) * (1 - sigmoid(x.dot(w1)))
    grad_w1 = grad_y_pred.T.dot(grad_final_sigmoid).dot(w2.T).dot(grad_first_sigmoid.T).dot(x)
    print grad_y_pred.shape
    print grad_final_sigmoid.shape
    print w2.shape
    print grad_first_sigmoid.shape
    print x.shape

    # Update weights - via batch learning
    w2 = w2 - learning_rate * grad_w2
    w1 = w1 - learning_rate * grad_w1





# ------- END OF THE CODE --------