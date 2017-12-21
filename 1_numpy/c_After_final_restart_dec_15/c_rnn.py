import copy, numpy as np,sys
np.random.seed(1234)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# 0. Data preprocessing and hyper parameter training dataset generation
int2binary = {}
binary_dim = 4

largest_number = pow(2,binary_dim)

binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
binary = binary[:,4:]

for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize neural network weights - these are the weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

print(synapse_0.shape)
print(synapse_1.shape)
print(synapse_h.shape)

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

number_of_epoch = 1
# 1. Loop training logic
for j in range(number_of_epoch):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    print(a)
    print(b)
    print(c,'\n--------\n')

    # Func: Forward Feed layer and operation
    for i in range(len(a)-1,-1,-1):
        
        temp_input = np.array([a[i],b[i]])
        y = c[i]

        state_1 = np.dot(temp_input,synapse_0) + np.dot(layer_1_values[0],synapse_h)
        state_1_act = sigmoid(state_1)
        layer_1_values.append(copy.deepcopy(state_1_act))

        state_2 = np.dot(state_1_act,synapse_1_update)
        state_2_act = sigmoid(state_2)

        layer_2_error = y - state_2_act
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(state_2))
        overallError += np.abs(layer_2_error[0])

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).dot(layer_2_delta.T)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta


    # print out progress
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")













































sys.exit()
# Fund: Update the functions
synapse_0 += synapse_0_update * alpha
synapse_1 += synapse_1_update * alpha
synapse_h += synapse_h_update * alpha    

synapse_0_update *= 0
synapse_1_update *= 0
synapse_h_update *= 0

sys.exit()
# moving along the positions in the binary encoding - this is the forward feed layers
for position in range(binary_dim):
    
    # generate input and output
    X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
    y = np.array([[c[binary_dim - position - 1]]]).T

    # hidden layer (input ~+ prev_hidden)
    layer_1 = np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h)
    layer_1 = sigmoid(layer_1)

    # output layer (new binary representation)
    layer_2 = np.dot(layer_1,synapse_1)
    layer_2 = sigmoid(layer_2)

    # did we miss?... if so, by how much?
    layer_2_error = y - layer_2
    layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
    overallError += np.abs(layer_2_error[0])

    # decode estimate so we can print it out
    d[binary_dim - position - 1] = np.round(layer_2[0][0])
    
    # store hidden layer so we can use it in the next timestep
    layer_1_values.append(copy.deepcopy(layer_1))

# Func: Capable of doing the stuffs
future_layer_1_delta = np.zeros(hidden_dim)
for position in range(binary_dim):
    
    X = np.array([[a[position],b[position]]])
    layer_1 = layer_1_values[-position-1]
    prev_layer_1 = layer_1_values[-position-2]
    
    # error at output layer
    layer_2_delta = layer_2_deltas[-position-1]
    # error at hidden layer
    layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

    # let's update all our weights so we can try again
    synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
    synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
    synapse_0_update += X.T.dot(layer_1_delta)
    
    future_layer_1_delta = layer_1_delta


synapse_0 += synapse_0_update * alpha
synapse_1 += synapse_1_update * alpha
synapse_h += synapse_h_update * alpha    

synapse_0_update *= 0
synapse_1_update *= 0
synapse_h_update *= 0

# print out progress
if(j % 1000 == 0):
    print("Error:" + str(overallError))
    print("Pred:" + str(d))
    print("True:" + str(c))
    out = 0
    for index,x in enumerate(reversed(d)):
        out += x*pow(2,index)
    print(str(a_int) + " + " + str(b_int) + " = " + str(out))
    print("------------")



sys.exit()


# 1. Loop training logic
for j in range(number_of_epoch):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding - this is the forward feed layers
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h)
        layer_1 = sigmoid(layer_1)

        # output layer (new binary representation)
        layer_2 = np.dot(layer_1,synapse_1)
        layer_2 = sigmoid(layer_2)

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    # Func: Capable of doing the stuffs
    future_layer_1_delta = np.zeros(hidden_dim)
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")




#  -- end code --

        