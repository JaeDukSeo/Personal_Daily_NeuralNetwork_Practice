import sklearn
from sklearn.datasets import make_classification,make_blobs
from sklearn import model_selection
import matplotlib.pyplot as plt
import minpy.numpy as np
from minpy.context import  gpu,cpu

with gpu(0):

    def sigmoid(x):
        return 1 / ( 1 - np.exp(-1*x))

    def d_sigmoid(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def tanh(x):
        return np.tanh(x)

    def d_tanh(x):
        return 1 - tanh(x)**2



    # 0. Seed the random value for repeatable
    np.random.seed(1234)

    # 1. Create a data set for the network to work on
    Data, Label = make_blobs(n_samples=1500, random_state=8,centers =2)

    Data = np.array(Data)
    Label = np.array(Label)

    # plt.scatter(Data[:,0],Data[:,1],c=Label)
    # plt.show()

    # 1.3 Split the training and test set
    data_x,data_y,label_x,label_y = model_selection.train_test_split(Data,Label,test_size =0.3333333)


    # 2. Declare hyper parameter

    number_of_epoch = 10
    learning_rate = 100

    w1 = np.random.randn(2,100)
    w2 = np.random.randn(100,150)
    w3 = np.random.randn(150,1)

    v1,v2,v3 = 0,0,0

    # 2.8 Open a figure
    plt.figure()

    # 3. Create the model and train
    for iter in range(number_of_epoch):
        
        current_x,current_label = sklearn.utils.shuffle(data_x,label_x)

        for i in range(0,current_x.shape[0],100):
            
            current_x_batch = current_x[i:i+100]
            current_label_batch = np.expand_dims(current_label[i:i+100],axis=1)

            layer_1 = current_x_batch.dot(w1)
            layer_1_act = tanh(layer_1)

            layer_2 = layer_1_act.dot(w2)
            layer_2_act = tanh(layer_2)

            layer_3 = layer_2_act.dot(w3)
            layer_3_act = sigmoid(layer_3)        

            cost = np.square(current_label_batch - layer_3_act) / (current_x.shape[0] * 2) 

            grad_3_part_1 = current_label_batch - layer_3_act
            grad_3_part_2 = d_sigmoid(layer_3)
            grad_3_part_3 = layer_2_act
            grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

            grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
            grad_2_part_2 = d_tanh(layer_2)
            grad_2_part_3 = layer_1_act
            grad_2  =  grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)    

            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
            grad_1_part_2 = d_tanh(layer_1)
            grad_1_part_3 = current_x_batch
            grad_1  =  grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)         

            w3= w3- learning_rate * grad_3
            w2 =w2- learning_rate * grad_2
            w1 =w1- learning_rate * grad_1
            
        # 3.5 Step wise learning rate decay
        # if iter == 80 : learning_rate = 0.1

        # 3.6 Test set try
        layer_1 = data_y.dot(w1)
        layer_1_act = tanh(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = tanh(layer_2)

        layer_3 = layer_2_act.dot(w3)
        layer_3_act = sigmoid(layer_3)  
        
        temp = data_y.asnumpy()
        layer_3_act = np.squeeze(layer_3_act.asnumpy()  )

        # 3.7 Make the plot
        plt.subplot(211)
        plt.title('Predicted : '+str(iter))
        plt.scatter(temp[:,0],temp[:,1],c=list(layer_3_act) )

        plt.subplot(212)
        plt.title('Ground Truth : '+str(iter))
        plt.scatter(temp[:,0],temp[:,1],c=label_y.asnumpy() )

        plt.pause(0.004)    


with cpu():

    def sigmoid(x):
        return 1 / ( 1 - np.exp(-1*x))

    def d_sigmoid(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def tanh(x):
        return np.tanh(x)

    def d_tanh(x):
        return 1 - tanh(x)**2



    # 0. Seed the random value for repeatable
    np.random.seed(1234)

    # 1. Create a data set for the network to work on
    Data, Label = make_blobs(n_samples=1500, random_state=8,centers =2)

    Data = np.array(Data)
    Label = np.array(Label)

    # plt.scatter(Data[:,0],Data[:,1],c=Label)
    # plt.show()

    # 1.3 Split the training and test set
    data_x,data_y,label_x,label_y = model_selection.train_test_split(Data,Label,test_size =0.3333333)


    # 2. Declare hyper parameter

    number_of_epoch = 10
    learning_rate = 100

    w1 = np.random.randn(2,100)
    w2 = np.random.randn(100,150)
    w3 = np.random.randn(150,1)

    v1,v2,v3 = 0,0,0

    # 2.8 Open a figure
    plt.figure()

    # 3. Create the model and train
    for iter in range(number_of_epoch):
        
        current_x,current_label = sklearn.utils.shuffle(data_x,label_x)

        for i in range(0,current_x.shape[0],100):
            
            current_x_batch = current_x[i:i+100]
            current_label_batch = np.expand_dims(current_label[i:i+100],axis=1)

            layer_1 = current_x_batch.dot(w1)
            layer_1_act = tanh(layer_1)

            layer_2 = layer_1_act.dot(w2)
            layer_2_act = tanh(layer_2)

            layer_3 = layer_2_act.dot(w3)
            layer_3_act = sigmoid(layer_3)        

            cost = np.square(current_label_batch - layer_3_act) / (current_x.shape[0] * 2) 

            grad_3_part_1 = current_label_batch - layer_3_act
            grad_3_part_2 = d_sigmoid(layer_3)
            grad_3_part_3 = layer_2_act
            grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

            grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
            grad_2_part_2 = d_tanh(layer_2)
            grad_2_part_3 = layer_1_act
            grad_2  =  grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)    

            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
            grad_1_part_2 = d_tanh(layer_1)
            grad_1_part_3 = current_x_batch
            grad_1  =  grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)         

            w3= w3- learning_rate * grad_3
            w2 =w2- learning_rate * grad_2
            w1 =w1- learning_rate * grad_1
            
        # 3.5 Step wise learning rate decay
        # if iter == 80 : learning_rate = 0.1

        # 3.6 Test set try
        layer_1 = data_y.dot(w1)
        layer_1_act = tanh(layer_1)

        layer_2 = layer_1_act.dot(w2)
        layer_2_act = tanh(layer_2)

        layer_3 = layer_2_act.dot(w3)
        layer_3_act = sigmoid(layer_3)  
        
        temp = data_y.asnumpy()
        layer_3_act = np.squeeze(layer_3_act.asnumpy()  )

        # 3.7 Make the plot
        plt.subplot(211)
        plt.title('Predicted : '+str(iter))
        plt.scatter(temp[:,0],temp[:,1],c=list(layer_3_act) )

        plt.subplot(212)
        plt.title('Ground Truth : '+str(iter))
        plt.scatter(temp[:,0],temp[:,1],c=label_y.asnumpy() )

        plt.pause(0.004)   
# ---- end code -----