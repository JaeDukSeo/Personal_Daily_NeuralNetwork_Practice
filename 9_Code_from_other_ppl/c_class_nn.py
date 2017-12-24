import numpy as np
import sys

def generate_dataset(output_dim = 8,num_examples=1000):
    def int2vec(x,dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    y_int = x_left_int + x_right_int

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]),int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x)
    y = np.array(y)
    
    return (x,y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_out2deriv(out):
    return out * (1 - out)

class Layer(object):
    
    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv):
        
        self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
    
    def forward(self,input):
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        return self.output
    
    def backward(self,output_delta):
        self.weight_output_delta = output_delta * self.nonlin_deriv(self.output)
        return self.weight_output_delta.dot(self.weights.T)
    
    def update(self,alpha=0.1):
        self.weights -= self.input.T.dot(self.weight_output_delta) * alpha

np.random.seed(1)

num_examples = 1000
output_dim = 12
iterations = 1000

x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

batch_size = 10
alpha = 0.1

input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])

layer_1 = Layer(input_dim,layer_1_dim,sigmoid,sigmoid_out2deriv)
layer_2 = Layer(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv)
layer_3 = Layer(layer_2_dim, output_dim,sigmoid, sigmoid_out2deriv)

for iter in range(iterations):
    error = 0

    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]  
        
        layer_1_out = layer_1.forward(batch_x)
        layer_2_out = layer_2.forward(layer_1_out)
        layer_3_out = layer_3.forward(layer_2_out)

        layer_3_delta = layer_3_out - batch_y
        layer_2_delta = layer_3.backward(layer_3_delta)
        layer_1_delta = layer_2.backward(layer_2_delta)
        layer_1.backward(layer_1_delta)
        
        # This is gradient update process 
        layer_1.update()
        layer_2.update()
        layer_3.update()
        
        error += (np.sum(np.abs(layer_3_delta * layer_3_out * (1 - layer_3_out))))

    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if(iter % 100 == 99):
        print("")




class DNI(object):
    
    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv,alpha = 0.1):
        
        # same as before
        self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv


        # new stuff
        self.weights_synthetic_grads = (np.random.randn(output_dim,output_dim) * 0.2) - 0.1
        self.alpha = alpha
    
    # used to be just "forward", but now we update during the forward pass using Synthetic Gradients :)
    def forward_and_synthetic_update(self,input):

    	# cache input
        self.input = input

        # forward propagate
        self.output = self.nonlin(self.input.dot(self.weights))
        
        # generate synthetic gradient via simple linear transformation
        self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads)

        # update our regular weights using synthetic gradient
        self.weight_synthetic_gradient = self.synthetic_gradient * self.nonlin_deriv(self.output)
        self.weights += self.input.T.dot(self.weight_synthetic_gradient) * self.alpha
        
        # return backpropagated synthetic gradient (this is like the output of "backprop" method from the Layer class)
        # also return forward propagated output (feels weird i know... )
        return self.weight_synthetic_gradient.dot(self.weights.T), self.output
    
    # this is just like the "update" method from before... except it operates on the synthetic weights
    def update_synthetic_weights(self,true_gradient):
        self.synthetic_gradient_delta = self.synthetic_gradient - true_gradient 
        self.weights_synthetic_grads += self.output.T.dot(self.synthetic_gradient_delta) * self.alpha
        


# ---- end code ---