import numpy as np
import sklearn
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

def LReLu(matrix):
    safe =  (matrix >0) * 1.0
    maks  = (matrix <=0) * 0.001
    return (safe *matrix ) + (maks * matrix)

def d_LReLu(matrix):
    safe =  (matrix >0) * 1.0
    maks  = (matrix <=0) * 0.001
    return (safe ) + (maks )

def sigmoid(x):
    return 1/ (1 + np.exp(-1 * x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

np.random.seed(4)


# 0. Preprocess data
mnist = input_data.read_data_sets("../4_tensorflow/MNIST_data/", one_hot=True)
x, x_label, y, y_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# x = x.reshape(-1, 28, 28, 1)  # 28x28x1 input img
# y = y.reshape(-1, 28, 28, 1)  # 28x28x1 input img


# 0.5 Hyper Parameters
number_of_epoch = 100
learning_rate = 0.0001
gamma  = 0.003
drop_out_rate = 0.4
past_cost = 1

v4 = 0
v3 = 0
v2 = 0
v1 = 0


input_d, h1_d,h2_d,h3_d,out_d = 784,400,100,300,784
w1 = np.random.randn(input_d,h1_d)
w2 = np.random.randn(h1_d,h2_d)
w3 = np.random.randn(h2_d,h3_d)
w4 = np.random.randn(h3_d,out_d)

# 1. Make the operations
for iter in range(number_of_epoch):

    x = sklearn.utils.shuffle(x)

    for i in range(0,len(x),1000):

        current_x = x[i:i+1000]
        # current_label = x_label[i:i+1000]

        layer_1 = current_x.dot(w1)
        layer_1_act = sigmoid(layer_1)
        layer_1_mask = np.random.binomial(1, drop_out_rate, size=layer_1_act.shape)
        layer_1_drop = layer_1_mask * layer_1_act

        layer_2 = layer_1_drop.dot(w2)
        layer_2_act = LReLu(layer_2)
        layer_2_mask = np.random.binomial(1, drop_out_rate, size=layer_2_act.shape)
        layer_2_drop = layer_2_mask * layer_2_act
        
        layer_3 = layer_2_drop.dot(w3)
        layer_3_act = sigmoid(layer_3)
        layer_3_mask = np.random.binomial(1, drop_out_rate, size=layer_3_act.shape)
        layer_3_drop = layer_3_mask * layer_3_act

        final = layer_3_drop.dot(w4)
        final_act = LReLu(final)
        final_mask = np.random.binomial(1, drop_out_rate, size=final_act.shape)
        fianl_drop = final_mask * final_act

        # if iter %100 == 0:
        cost = np.square(fianl_drop - current_x).sum() / (2 * len(current_x))
        print "Current Epoch : ",iter, "  Current Batch : ",i,' ',i+1000,' cost :', cost 


        grad_4_part_1 = (fianl_drop - current_x) / len(current_x)
        grad_4_part_2 = d_LReLu(final)
        grad_4_part_3 = layer_3_drop
        grad_4      =grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

        grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
        grad_3_part_2 = d_sigmoid(layer_3)
        grad_3_part_3 = layer_2_drop
        grad_3      =grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

        grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_LReLu(layer_2)
        grad_2_part_3 = layer_1_drop
        grad_2       = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = current_x
        grad_1 =     grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)    


        v4 = gamma * v4 +     learning_rate* grad_4
        v3 = gamma * v3 +     learning_rate* grad_3
        v2 = gamma * v2 +     learning_rate* grad_2
        v1 = gamma * v1 +     learning_rate* grad_1
        w4-=v4
        w3-=v3
        w2-=v2
        w1-=v1

        # w4-=learning_rate* grad_4
        # w3-=learning_rate* grad_3
        # w2-=learning_rate* grad_2
        # w1-=learning_rate *grad_1

    if iter == 5:
        learning_rate = 0.0000001
    if iter == 15:
        learning_rate = 0.00000001


    layer_1 = y.dot(w1)
    layer_1_act = sigmoid(layer_1)
    layer_1_mask = np.random.binomial(1, 1.0, size=layer_1_act.shape)
    layer_1_drop = layer_1_mask * layer_1_act

    layer_2 = layer_1_drop.dot(w2)
    layer_2_act = LReLu(layer_2)
    layer_2_mask = np.random.binomial(1, 1.0, size=layer_2_act.shape)
    layer_2_drop = layer_2_mask * layer_2_act

    layer_3 = layer_2_drop.dot(w3)
    layer_3_act = sigmoid(layer_3)
    layer_3_mask = np.random.binomial(1, 1.0, size=layer_3_act.shape)
    layer_3_drop = layer_3_mask * layer_3_act

    final = layer_3_drop.dot(w4)
    final_act = LReLu(final)
    final_mask = np.random.binomial(1, 1.0, size=final_act.shape)
    fianl_drop = final_mask * final_act

    cost = np.square(fianl_drop - y).sum() / (2 * len(y))
    print '---------------------------------'
    print "Current Epoch : ",iter,'  test set cost :', cost ,"  Rate of Chnage : ",((past_cost- cost)/past_cost) * 100
    print '---------------------------------'
    past_cost = cost

    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))

    start = 0
    for i in range(n):
        # Display original images
        for j in range(n):
            # Draw the generated digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = y[start + j].reshape(28,28)

        for j in range(n):
            # Draw the generated digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = final_act[start + j].reshape(28,28)
        
        start += 4

    print("Original Images")     
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.savefig('z_auto_nov_10_practice_2/'+str(iter) + '_OG.png', bbox_inches='tight')
    # plt.show()

    # print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.savefig('z_auto_nov_10_practice_2/'+str(iter) + '_predict.png', bbox_inches='tight')







# ------- END CODE ---- 873658304.74   48957195.6381
#                       366027321.051  317157370.495