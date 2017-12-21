import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

np.random.seed(1345367)


def LReLU(matrix):
    safe = (matrix>0) * 1.0
    mask = (matrix<=0) * 0.001
    return (safe * matrix) +  (mask * matrix)

def d_LReLU(matrix):
    safe = (matrix>0) * 1.0
    mask = (matrix<=0) * 0.001
    return (safe ) +  (mask )   

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 0. Preprocess data
mnist = input_data.read_data_sets("../4_tensorflow/MNIST_data/", one_hot=True)
x, x_label, y, y_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# x = x.reshape(-1, 28, 28, 1)  # 28x28x1 input img
# y = y.reshape(-1, 28, 28, 1)  # 28x28x1 input img

# 0.5 hyper parameter
input_d,h1_d,h2_d,h3_d,out_d  = 784,250,100,250,784

w1 = np.random.rand(input_d,h1_d)
w2 = np.random.rand(h1_d,h2_d)
w3 = np.random.rand(h2_d,h3_d)
w4 = np.random.rand(h3_d,out_d)

learning_rate = 0.000000000000000003
number_of_epoch = 1600
past_i = 0

for iter in range(number_of_epoch):
    for i in range(1000,len(x),1000):

        current_x = x[past_i:i]

        layer_1 = current_x.dot(w1)
        layer_1_act = LReLU(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = LReLU(layer_2)

        layer_3 = layer_2_act.dot(w3)
        layer_3_act = LReLU(layer_3)

        final = layer_3_act.dot(w4)
        final_act = LReLU(final)

        loss = (final - current_x).sum() / float(len(current_x))
        print "Current Epoch: ",iter, "  Current Error : ",loss, " Current I : ",past_i ,' : ',i
        past_i = i
        
        grad_4_part_1 = (final - current_x) 
        grad_4_part_2 = d_LReLU(final)
        grad_4_part_3 = layer_3_act
        grad_4 = grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

        grad_3_part_1 = (grad_4_part_1* grad_4_part_2).dot(w4.T)
        grad_3_part_2 = d_LReLU(layer_3)
        grad_3_part_3 = layer_2_act
        grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

        grad_2_part_1 = (grad_3_part_1*grad_3_part_2).dot(w3.T)
        grad_2_part_2 = d_LReLU(layer_2)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1* grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_LReLU(layer_1)
        grad_1_part_3 = current_x
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w1 -= learning_rate * grad_1
        w2 -= learning_rate * grad_2
        w3 -= learning_rate * grad_3
        w4 -= learning_rate * grad_4

    past_i = 0
    if iter > 1000:
        learning_rate = 0.00000000000000000008
        
        
        

    if iter % 100 == 0 :

        current_y = y[:16]

        layer_1 = current_y.dot(w1)
        layer_1_act = LReLU(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = LReLU(layer_2)

        layer_3 = layer_2_act.dot(w3)
        layer_3_act = LReLU(layer_3)

        final = layer_3_act.dot(w4)
        final_act = LReLU(final)


        n = 4
        canvas_orig = np.empty((28 * n, 28 * n))
        canvas_recon = np.empty((28 * n, 28 * n))


        start = 0
        for i in range(n):
            # Display original images
            for j in range(n):
                # Draw the generated digits
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = current_y[start + j].reshape(28,28)

            for j in range(n):
                # Draw the generated digits
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = final_act[start + j].reshape(28,28)
            
            print start
            start += 4
            print start
            

        print("Original Images")     
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.savefig('img/'+str(iter) + '_OG.png', bbox_inches='tight')
        # plt.show()

        # print("Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.savefig('img/'+str(iter) + '_predict.png', bbox_inches='tight')
        
        # plt.show()




# -------- END CODE ---------