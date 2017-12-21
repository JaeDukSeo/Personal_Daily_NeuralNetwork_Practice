import numpy as np

def sigmoid(x):
    return 1 / (  1 + np.exp(-1*x) )

def d_sigmoid(x):
    return sigmoid(x) * ( 1 - sigmoid(x))

np.random.seed(1)
X = np.array(
            [[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

input_d,h1_d,h2_d,out_d = 3,4,4,1
number_of_epoch = 60000
learn_rates = [0.001,0.01,0.1,1,10,100,1000]




for current_rate in learn_rates:

    print "\nTraining With Alpha:" + str(current_rate)
    w1 = 2*np.random.random((3,4)) - 1
    w2 = 2*np.random.random((4,1)) - 1
    
    for j in xrange(60000):

        layer_1 = X.dot(w1)
        layer_1_act = sigmoid(layer_1)

        final = layer_1_act.dot(w2)
        final_act = sigmoid(final)

        if (j% 10000) == 0:
            print "Error after "+str(j)+" iterations:" + str(np.mean(np.abs(final_act-y)))

        grad_2_part_1 = final_act - y
        grad_2_part_2 = d_sigmoid(final)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = X
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w2 = w2 - current_rate*grad_2
        w1 = w1 - current_rate*grad_1

    
        
w1_1 = 2*np.random.random((3,4)) - 1
w2_1 = 2*np.random.random((4,1)) - 1

w1_2 = np.random.random((3,4)) - 1
w2_2 = np.random.random((4,1)) - 1

w1_3 = np.random.randn(3,4)
w2_3 = np.random.randn(4,1)

w1_4 = np.random.rand(3,4)
w2_4 = np.random.rand(4,1)
    
weight_list = [ [w1_1,w2_1],[w1_2,w2_2],[w1_3,w2_3],[w1_4,w2_4]   ]

for current_weight in weight_list:

    print "\nTraining With Weight:" + str(current_weight)

    w1 = current_weight[0]
    w2 = current_weight[1]
    
    for j in xrange(60000):

        layer_1 = X.dot(w1)
        layer_1_act = sigmoid(layer_1)

        final = layer_1_act.dot(w2)
        final_act = sigmoid(final)

        if (j% 10000) == 0:
            print "Error after "+str(j)+" iterations:" + str(np.mean(np.abs(final_act-y)))

        grad_2_part_1 = final_act - y
        grad_2_part_2 = d_sigmoid(final)
        grad_2_part_3 = layer_1_act
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_sigmoid(layer_1)
        grad_1_part_3 = X
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        w2 = w2 - 1*grad_2
        w1 = w1 - 1*grad_1



for current_rate in learn_rates:

    for current_weight in weight_list:

        print "\nTraining With Weight:" + str(current_weight)
        print "\nTraining With Laerning:" + str(current_rate)
        

        w1 = current_weight[0]
        w2 = current_weight[1]
        
        for j in xrange(60000):

            layer_1 = X.dot(w1)
            layer_1_act = sigmoid(layer_1)

            final = layer_1_act.dot(w2)
            final_act = sigmoid(final)

            if (j% 10000) == 0:
                print "Error after "+str(j)+" iterations:" + str(np.mean(np.abs(final_act-y)))

            grad_2_part_1 = final_act - y
            grad_2_part_2 = d_sigmoid(final)
            grad_2_part_3 = layer_1_act
            grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
            grad_1_part_2 = d_sigmoid(layer_1)
            grad_1_part_3 = X
            grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

            w2 = w2 - current_rate*grad_2
            w1 = w1 - current_rate*grad_1



# --------- END OF THE CODE