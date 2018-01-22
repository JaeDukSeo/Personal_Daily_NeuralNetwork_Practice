
import numpy as np,sys,time
from sklearn import datasets
from sklearn.datasets import make_classification

np.random.seed(4567)


def log(x):
    return 1  / ( 1 + np.exp(-1*x))
def d_log(x):
    return log(x) * ( 1- log(x))

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - tanh(x) ** 2

def IDEN(x):
    return x
def d_IDEN(x):
    return 1


def generate_dataset(output_dim = 8,num_examples=1000):
    def int2vec(x,dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    print(x_left_int[0])
    print(x_right_int[0])
    y_int = x_left_int + x_right_int
    print(y_int[0])

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


digits = datasets.load_digits()
image = np.reshape(digits.images.T,(64,-1)).T
label = digits.target   


# 0. Generate Training Data
X, Y = make_classification(n_samples=500,n_features=64,
                        class_sep=4, n_repeated = 0,n_redundant=0, n_classes=2,
                        n_informative=2,n_clusters_per_class=2)
# plt.scatter(X[:,0],X[:,1],c = Y)
# plt.show()
Y = np.expand_dims(Y,axis=1)
image = X
label = Y


num_epoch = 1000
learning_rate = 0.000001

w1 = np.random.randn(64,6)
w2 = np.random.randn(6,18)
w3 = np.random.randn(18,1)

w1_syth = np.random.randn(6,6)
w2_syth = np.random.randn(18,18)
w3_syth = np.random.randn(1,1)

w1_nn,w2_nn,w3_nn = w1,w2,w3
w1_DN,w2_DN,w3_DN = w1,w2,w3
w1_nn_noise,w2_nn_noise,w3_nn_noise = w1,w2,w3
w1_DN_noise,w2_DN_noise,w3_DN_noise = w1,w2,w3




for iter in range(num_epoch):
    
    layer_1 = X.dot(w1)
    layer_1_act = tanh(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = log(layer_2)

    layer_3 = layer_2_act.dot(w3)
    layer_3_act = log(layer_3)

    cost = np.square(layer_3_act - label).sum() * 0.5
    print("Current Iter : ", iter, " current cost : ", cost,end="\r")

    grad_3_part_1 = layer_3_act - label
    grad_3_part_2 = d_log(layer_3)
    grad_3_part_3 = layer_2_act
    grad_3 =    grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
    grad_2_part_2 = d_log(layer_2)
    grad_2_part_3 = layer_1_act
    grad_2 =    grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

    grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
    grad_1_part_2 = d_tanh(layer_1)
    grad_1_part_3 = X
    grad_1 =    grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

    w3 = w3 - learning_rate * grad_3
    w3 = w3 - learning_rate * grad_3
    w3 = w3 - learning_rate * grad_3
    











print("\n-----------------------")









for iter in range(num_epoch):
    
    # ------ Normal Layer 1 ---------
    layer_1 = image.dot(w1_DN)
    layer_1_act = tanh(layer_1)
    # ------ Normal Layer 1 ---------
    # ------ Sythn Layer 1 ---------
    syth_grad_1_part_1 = layer_1_act.dot(w1_syth)
    syth_grad_1_part_2 = d_tanh(layer_1)
    syth_grad_1_part_3 = image
    syth_grad_1 =   syth_grad_1_part_3.T.dot(syth_grad_1_part_1 * syth_grad_1_part_2)

    w1_DN = w1_DN - learning_rate * syth_grad_1
    # ------ Sythn Layer 1 ---------
    

    # ------ Normal Layer 2 ---------
    layer_2 = layer_1_act.dot(w2_DN)
    layer_2_act = log(layer_2)
    # ------ Normal Layer 2 ---------
    # ------ Sythn Layer  2---------
    syth_grad_2_part_1 = layer_2_act.dot(w2_syth)
    syth_grad_2_part_2 = d_log(layer_2)
    syth_grad_2_part_3 = layer_1_act
    syth_grad_2 =   syth_grad_2_part_3.T.dot(syth_grad_2_part_1 * syth_grad_2_part_2)

    w2_DN = w2_DN - learning_rate * syth_grad_2
    layer_2_delta = (syth_grad_2_part_1 * syth_grad_2_part_2).dot(w2_DN.T)
    # ------ Sythn Layer  2 ---------
    # ------ Layer 1 Groud Truth Weight Update -----
    syth_gt_grad_1_part_1 = syth_grad_1_part_1 - layer_2_delta
    syth_gt_grad_1_part_2 = layer_1_act
    syth_gt_grad_1 = syth_gt_grad_1_part_2.T.dot(syth_gt_grad_1_part_1)
    w1_syth = w1_syth - learning_rate * syth_gt_grad_1
    # ------ Layer 1 Groud Truth Weight Update -----


    # ------ Normal Layer 3 ---------
    layer_3 = layer_2_act.dot(w3_DN)
    layer_3_act = log(layer_3)
    # ------ Normal Layer 3 ---------
    # ------ Sythn Layer  3---------
    syth_grad_3_part_1 = layer_3_act.dot(w3_syth)
    syth_grad_3_part_2 = d_log(layer_3)
    syth_grad_3_part_3 = layer_2_act
    syth_grad_3 =   syth_grad_3_part_3.T.dot(syth_grad_3_part_1 * syth_grad_3_part_2)

    w3_DN = w3_DN - learning_rate * syth_grad_3
    layer_3_delta = (syth_grad_3_part_1 * syth_grad_3_part_2).dot(w3_DN.T)
    # ------ Sythn Layer  3 ---------
    # ------ Layer 2 Groud Truth Weight Update -----
    syth_gt_grad_2_part_1 = syth_grad_2_part_1 - layer_3_delta
    syth_gt_grad_2_part_2 = layer_2_act
    syth_gt_grad_2 = syth_gt_grad_2_part_2.T.dot(syth_gt_grad_2_part_1)
    w2_syth = w2_syth - learning_rate * syth_gt_grad_2
    # ------ Layer 3 Groud Truth Weight Update -----

    cost = np.square(layer_3_act - label).sum() * 0.5
    print("Current Iter : ", iter, " current cost : ", cost,end="\r")

    # ------ Layer 3 Groud Truth Weight Update -----
    syth_gt_grad_3_part_1 = syth_grad_3_part_1 - cost
    syth_gt_grad_3_part_2 = layer_3_act
    syth_gt_grad_3 = syth_gt_grad_3_part_2.T.dot(syth_gt_grad_3_part_1)
    w3_syth = w3_syth - learning_rate * syth_gt_grad_3
    # ------ Layer 3 Groud Truth Weight Update -----

print("\n-----------------------")


layer_1 = X.dot(w1)
layer_1_act = tanh(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = log(layer_2)

layer_3 = layer_2_act.dot(w3)
layer_3_act = log(layer_3)

cost = np.square(layer_3_act - label).sum() * 0.5
print(cost)



class DNI(object):
    
    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv,alpha ):
        
        self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        self.weights_synthetic_grads = (np.random.randn(output_dim,output_dim) * 0.2) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        self.alpha = alpha
    
    def forward_and_synthetic_update(self,input):
        
        # Traditional Forward Feed Process
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        
        # 
        self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads)
        self.weight_synthetic_gradient = self.synthetic_gradient * self.nonlin_deriv(self.output)
        self.weights += self.input.T.dot(self.weight_synthetic_gradient) * self.alpha
        
        return self.weight_synthetic_gradient.dot(self.weights.T), self.output
    
    def update_synthetic_weights(self,true_gradient):
        self.synthetic_gradient_delta = self.synthetic_gradient - true_gradient
        self.weights_synthetic_grads += self.output.T.dot(self.synthetic_gradient_delta) * self.alpha



print("\n----------------------")
print("\n----------------------")



layer_1 = DNI(64,6,sigmoid,sigmoid_out2deriv,learning_rate)
layer_2 = DNI(6,18,sigmoid,sigmoid_out2deriv,learning_rate)
layer_3 = DNI(18,1,sigmoid, sigmoid_out2deriv,learning_rate)

for iter in range(num_epoch):
    error = 0
    
    _, layer_1_out = layer_1.forward_and_synthetic_update(X)
    layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
    layer_1.update_synthetic_weights(layer_1_delta)
    
    layer_2_delta, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)
    layer_2.update_synthetic_weights(layer_2_delta)

    cost = np.square(layer_3_out - Y).sum() * 0.5
    print("Current Iter : ", iter, " current cost : ", cost,end="\r")

    layer_3_delta =  layer_3_out - Y
    layer_3.update_synthetic_weights(layer_3_delta)





_, layer_1_out = layer_1.forward_and_synthetic_update(X)
layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
layer_2_delta, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)

print("\n----------------------")
cost = np.square(layer_3_out - label).sum() * 0.5
print(cost)






sys.exit()


num_examples = 1000
output_dim = 12
iterations = 2000

x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

batch_size = 1000
alpha = 0.03


class DNI(object):
    
    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv,alpha ):
        
        self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        self.weights_synthetic_grads = (np.random.randn(output_dim,output_dim) * 0.2) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        self.alpha = alpha
    
    def forward_and_synthetic_update(self,input):
        
        # Traditional Forward Feed Process
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        
        # 
        self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads)
        self.weight_synthetic_gradient = self.synthetic_gradient * self.nonlin_deriv(self.output)
        self.weights += self.input.T.dot(self.weight_synthetic_gradient) * self.alpha
        
        return self.weight_synthetic_gradient.dot(self.weights.T), self.output
    
    def update_synthetic_weights(self,true_gradient):
        self.synthetic_gradient_delta = self.synthetic_gradient - true_gradient
        self.weights_synthetic_grads += self.output.T.dot(self.synthetic_gradient_delta) * self.alpha
        
#  input = 24, output = 12 , layer_1_dim = 128, layer_2_dim = 64
start = time.time()
input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])
layer_1 = DNI(input_dim,layer_1_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_2 = DNI(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_3 = DNI(layer_2_dim, output_dim,sigmoid, sigmoid_out2deriv,alpha)


for iter in range(iterations):
    error = 0

    batch_x = x
    batch_y = y
    
    _, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
    layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
    layer_1.update_synthetic_weights(layer_1_delta)
    
    layer_2_delta, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)
    layer_2.update_synthetic_weights(layer_2_delta)

    layer_3_delta =  layer_3_out - batch_y
    layer_3.update_synthetic_weights(layer_3_delta)

    error += (np.sum(np.abs(layer_3_delta * layer_3_out * (1 - layer_3_out))))

    if(error < 0.1):
        sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
        break       
        
    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if(iter % 100 == 0):
        print("")

end = time.time()


_, layer_1_out = layer_1.forward_and_synthetic_update(x)
layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
layer_2_delta, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)

for iter in range(10):
    print(x[iter][:12].dot(2**np.arange(x[iter][:12].size)[::-1])  )
    print(x[iter][12:].dot(2**np.arange(x[iter][:12].size)[::-1])  )
    print("-----------")
    print(layer_3_out[iter].dot(2**np.arange(x[iter][:12].size)[::-1]))

    truteh = x[iter][:12] + x[iter][12:]
    print("The truth data: ",truteh.dot(2**np.arange(x[iter][:12].size)[::-1]),'\n')
    
print("\n\n------------\nTraining Time: ",end - start )
