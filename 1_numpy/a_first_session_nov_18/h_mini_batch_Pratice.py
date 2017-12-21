import numpy as np ,sys
from sklearn.utils import shuffle
np.random.seed(1)

def sigmoid(x):
    return 1/(1+ np.exp(-1*x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 0. Data preprocessing and declare hyper parameter
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])
number_epoch = 300
input_d,h1_d,h2_d,out_d = 3,8,6,1


w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,out_d)
lr = 10

# 0.5 Shuffle the data
shuffle_x,shuffle_y = shuffle(x,y)


layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

final = layer_2_act.dot(w3)
final_act = sigmoid(final)
print final_act


# 1. declare operations
for i in range(number_epoch):

    for i,j in ((0,2),(2,4)):

        current_y = shuffle_y[i:j]
        current_x = shuffle_x[i:j]

        layer_1 = current_x.dot(w1)
        layer_1_act = sigmoid(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = sigmoid(layer_2)

        final = layer_2_act.dot(w3)
        final_act = sigmoid(final)

        if i%100 == 0:
            print "Error Rate: ",(final_act-current_y).sum()

        grad_3_part_1 = final_act-current_y
        grad_3_part_2 = d_sigmoid(final)
        grad_3_part_3 = layer_2_act
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_sigmoid(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = current_x
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w3 -= lr*grad_3
        w2 -= lr*grad_2
        w1 -= lr*grad_1






layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

final = layer_2_act.dot(w3)
final_act = sigmoid(final)
print final_act




sys.exit()
# -------------- SGD ONE LINE BATCH -------------------

layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

final = layer_2_act.dot(w3)
final_act = sigmoid(final)
print final_act


# 1. declare operations
for i in range(number_epoch):

    for x_current in range(0,len(x)):

        x_expand =np.expand_dims(x[x_current],axis=1)

        layer_1 = x_expand.T.dot(w1)
        layer_1_act = sigmoid(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = sigmoid(layer_2)

        final = layer_2_act.dot(w3)
        final_act = sigmoid(final)

        if i%100 == 0:
            print "Error Rate: ",(final_act-y[x_current])

        grad_3_part_1 = final_act-y[x_current]
        grad_3_part_2 = d_sigmoid(final)
        grad_3_part_3 = layer_2_act
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)
        
        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_sigmoid(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = x_expand
        grad_1 = grad_1_part_3.dot(grad_1_part_1 * grad_1_part_2)

        w3 -= grad_3
        w2 -= grad_2
        w1 -= grad_1
        


layer_1 = x.dot(w1)
layer_1_act = sigmoid(layer_1)

layer_2 = layer_1_act.dot(w2)
layer_2_act = sigmoid(layer_2)

final = layer_2_act.dot(w3)
final_act = sigmoid(final)  
print final_act


w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,out_d)

learning_rate = [0.001,0.01,0.1,1,10,100,1000000]

for lr in learning_rate:

    print "Current Learing Rate: ",lr

# 1. declare operations
    for i in range(number_epoch):
        layer_1 = x.dot(w1)
        layer_1_act = sigmoid(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = sigmoid(layer_2)

        final = layer_2_act.dot(w3)
        final_act = sigmoid(final)

        if i%100 == 0:
            print "Error Rate: ",(final_act-y).sum()

        grad_3_part_1 = final_act-y
        grad_3_part_2 = d_sigmoid(final)
        grad_3_part_3 = layer_2_act
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_sigmoid(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = x
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w3 -= lr*grad_3
        w2 -= lr*grad_2
        w1 -= lr*grad_1


    layer_1 = x.dot(w1)
    layer_1_act = sigmoid(layer_1)

    layer_2 = layer_1_act.dot(w2)
    layer_2_act = sigmoid(layer_2)

    final = layer_2_act.dot(w3)
    final_act = sigmoid(final)  
    print final_act



# -------- END OF THE CDOE --------