from sklearn.datasets import make_classification,make_moons
from sklearn.utils import shuffle
import matplotlib.pyplot as plt, numpy as np,sys,sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics

np.random.seed(4567890)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1.0 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def d_sgimoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 0. Data Preprocess and declare hyper parameter
X, Y = make_classification(n_samples=1800,n_features=2,
                        # n_classes = 3,
                        class_sep=0.45, n_redundant=0, 
                        n_informative=2,
                        n_clusters_per_class=1)

X,Y = make_moons(n_samples=1500, random_state=20, noise=0.035)
x_data, y_data, x_label_og, y_label_og = train_test_split(X, Y, train_size=0.833333 ,random_state=30)

# plt.scatter(x_data[:,0],x_data[:,1],c=x_label_og)
# plt.show()
# plt.scatter(y_data[:,0],y_data[:,1],c=y_label_og)
# plt.show()

input_d,h1_d,h2_d,h3_d,out_d = 2,30 ,40,45,1

w1 = np.random.randn(input_d,h1_d) +34312
w2 = np.random.randn(h1_d,h2_d) +34312
w3 = np.random.randn(h2_d,h3_d)+34312
w4 = np.random.randn(h3_d,out_d)+34312

number_of_epoch = 150
past_i = 0
learing_rates = [10000,1000,100,10,1,0,0.1,0.01,0.001,0.00000001,0.0000001]
added_values =[3423,67,5,7,0,0, 0.4 ,0.056,0.00342,0.00005431,0.00005435431]

for learning_rate in range(0,len(learing_rates)):

    for iter in range(number_of_epoch):

        x_data,x_label_og = sklearn.utils.shuffle(x_data,x_label_og)

        for i in range(250,1500,250):

            current_x_data = x_data[past_i:i]
            current_y_data = np.expand_dims(x_label_og[past_i:i],axis=1)

            layer_1  = current_x_data.dot(w1)
            layer_1_act  = tanh(layer_1)

            layer_2  = layer_1_act.dot(w2)
            layer_2_act  = tanh(layer_2)

            layer_3  = layer_2_act.dot(w3)
            layer_3_act  = tanh(layer_3)

            final  = layer_3_act.dot(w4)
            final_act  = sigmoid(final)
            
            loss = np.square(final_act - current_y_data).sum() / ( 2 * len(current_y_data))
            print "Loss : ",loss,' batch : ',past_i,' ',i
            past_i = i

            grad_4_part_1 = (final_act - current_y_data)
            grad_4_part_2 = d_sgimoid(final)
            grad_4_part_3 = layer_3_act
            grad_4 =  grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)
            
            grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
            grad_3_part_2 =d_tanh(layer_3)
            grad_3_part_3 =layer_2_act
            grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

            grad_2_part_1 =(grad_3_part_1 * grad_3_part_1).dot(w3.T)
            grad_2_part_2 =d_tanh(layer_2)
            grad_2_part_3 =layer_1_act
            grad_2 =     grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

            grad_1_part_1 =(grad_2_part_1 * grad_2_part_1).dot(w2.T)
            grad_1_part_2 =d_tanh(layer_1)
            grad_1_part_3 =current_x_data
            grad_1 =     grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

            w4 -= learing_rates[learning_rate]*grad_4
            w3 -= learing_rates[learning_rate]*grad_3
            w2 -= learing_rates[learning_rate]*grad_2
            w1 -= learing_rates[learning_rate]*grad_1

        layer_1  = y_data.dot(w1)
        layer_1_act  = tanh(layer_1)

        layer_2  = layer_1_act.dot(w2)
        layer_2_act  = tanh(layer_2)

        layer_3  = layer_2_act.dot(w3)
        layer_3_act  = tanh(layer_3)

        final  = layer_3_act.dot(w4)
        final_act  = np.round(sigmoid(final))

        loss = (final_act - y_label_og).sum() 
        print "epoch : ",iter,'  loss: ',loss, " Learning Rate : ",learning_rate
        print metrics.accuracy_score(y_label_og,final_act),'\n\n'
        past_i =0

    layer_1  = y_data.dot(w1)
    layer_1_act  = tanh(layer_1)

    layer_2  = layer_1_act.dot(w2)
    layer_2_act  = tanh(layer_2)

    layer_3  = layer_2_act.dot(w3)
    layer_3_act  = tanh(layer_3)

    final  = layer_3_act.dot(w4)
    final_act  = np.round(np.squeeze(sigmoid(final)))

    print "Leanring Rate : ",learing_rates[learning_rate]
    print metrics.accuracy_score(y_label_og,final_act)
    print metrics.confusion_matrix(y_label_og,final_act)

    plt.scatter(y_data[:,0],y_data[:,1],c=final_act)
    plt.show()
    plt.scatter(y_data[:,0],y_data[:,1],c=y_label_og)
    plt.show()

    w1 = np.random.randn(input_d,h1_d) + added_values[learning_rate]
    w2 = np.random.randn(h1_d,h2_d) + added_values[learning_rate]
    w3 = np.random.randn(h2_d,h3_d)+ added_values[learning_rate]
    w4 = np.random.randn(h3_d,out_d)+ added_values[learning_rate]


# --- END ---