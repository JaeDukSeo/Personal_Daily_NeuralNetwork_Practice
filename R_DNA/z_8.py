import tensorflow as tf
import random
import sys
import numpy as np
from numpy import float32
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib as mpl

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth=True
# np.random.seed(678)
# tf.set_random_seed(768)

# Activation Functions - however there was no indication in the original paper
def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf.multiply(tf_log(x),tf.subtract(1.0,tf_log(x))) 
def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))
def tf_softmax(x): return tf.nn.softmax(x)

# -1. DNA encode table
protein_table = {"TTT" : "F", "CTT" : "L", "ATT" : "I", "GTT" : "V",
           "TTC" : "F", "CTC" : "L", "ATC" : "I", "GTC" : "V",
           "TTA" : "L", "CTA" : "L", "ATA" : "I", "GTA" : "V",
           "TTG" : "L", "CTG" : "L", "ATG" : "M", "GTG" : "V",
           "TCT" : "S", "CCT" : "P", "ACT" : "T", "GCT" : "A",
           "TCC" : "S", "CCC" : "P", "ACC" : "T", "GCC" : "A",
           "TCA" : "S", "CCA" : "P", "ACA" : "T", "GCA" : "A",
           "TCG" : "S", "CCG" : "P", "ACG" : "T", "GCG" : "A",
           "TAT" : "Y", "CAT" : "H", "AAT" : "N", "GAT" : "D",
           "TAC" : "Y", "CAC" : "H", "AAC" : "N", "GAC" : "D",
            "CAA" : "Q", "AAA" : "K", "GAA" : "E",
            "CAG" : "Q", "AAG" : "K", "GAG" : "E",
           "TGT" : "C", "CGT" : "R", "AGT" : "S", "GGT" : "G",
           "TGC" : "C", "CGC" : "R", "AGC" : "S", "GGC" : "G",
            "CGA" : "R", "AGA" : "R", "GGA" : "G",
           "TGG" : "W", "CGG" : "R", "AGG" : "R", "GGG" : "G" 
           }

# 0. Generate Random Data of dna to convert to Protein
num_training_data = 1000
length_of_protein = 1

save_location = 'plt8/'
proportion_rate = 2000
decay_rate = 0.008
compare_range = 100

learning_rate = 0.0009
# above safe
learning_rate = 0.003

beta_1 ,beta_2= 0.9, 0.999
adam_e = 0.00000001

num_epoch = 1201
print_size = 200
batch_size = 100


hidden_layers = 300

# 1. Array to Contain all of the data 
dna_data = np.array([])
protein_data = np.array([])
dna_data_real = np.array([])

for training_index in range(num_training_data):
    
    current_data = np.array([])
    protein_sequence = np.array([])
    
    for _ in range(length_of_protein):
        dna,protein = random.choice(list(protein_table.items()))
        dna_data_real = np.append(dna_data_real,dna)
        for d in dna:
            if   d == "A":
                current_data = np.append(current_data,np.array([0,0,0,1]).T)
            elif d == "C":
                current_data = np.append(current_data,np.array([0,0,1,0]).T)
            elif d == "G":
                current_data = np.append(current_data,np.array([0,1,0,0]).T)
            elif d == "T":
                current_data = np.append(current_data,np.array([1,0,0,0]).T)
        protein_sequence = np.append(protein_sequence,[protein])

    if training_index == 0 :
        dna_data = np.expand_dims(current_data,axis=0)
        protein_data = np.expand_dims(protein_sequence,axis=0)
    else:
        dna_data = np.vstack((dna_data,np.expand_dims(current_data,axis=0)))
        protein_data = np.vstack((protein_data,np.expand_dims(protein_sequence,axis=0)))
        
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(protein_data)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
protein_data = onehot_encoder.fit_transform(integer_encoded)

print('------ One Hot Encoded Protein Data-----')
print(dna_data.shape)
print(protein_data.shape)

# 2. Make the class
class FCNN:
    
    def __init__(self,input=None,output=None,act=None,d_act=None,random_seed =None):

        self.w = tf.Variable(tf.random_normal([input,output],seed=random_seed))

        self.activation = act
        self.d_activation = d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

        self.input,self.layer,self.layerA= None,None,None

    def getw(self): return self.w

    def feedforward(self,input=None):
        self.input = input
        self.layer  = tf.matmul(self.input,self.w)
        self.layerA = self.activation(self.layer)
        return self.layerA
    
    def backprop(self,gradient=None):
        
        grad_part_1 = gradient
        grad_part_2 = self.d_activation(self.layer) 
        grad_part_3 = self.input

        grad = tf.matmul(tf.transpose(grad_part_3),tf.multiply(grad_part_1,grad_part_2))
        grad_pass = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        assign = []
        assign.append(tf.assign(self.m,beta_1*self.m + (1.0-beta_1)*grad))
        assign.append(tf.assign(self.v,beta_2*self.v + (1.0-beta_2)*tf.square(grad) ))    

        m_hat = tf.divide(self.m, tf.subtract(1.0,beta_1) )
        v_hat = tf.divide(self.v, tf.subtract(1.0,beta_2) )

        adam_middle = tf.divide(learning_rate,tf.add(tf.sqrt(v_hat),adam_e))
        assign.append(  tf.assign(self.w, tf.subtract(self.w,tf.multiply(adam_middle,m_hat)))  )

        return grad_pass,assign


def auto_diff(compare_number=None,random_seed=None):
    
    tf.reset_default_graph()
    print('\n\n========= Starting Comparison : ',compare_number, ' ==== AUTO ========')
    
    # 3.Make the Layers
    layer1 = FCNN(12,hidden_layers,tf_log,d_tf_log,random_seed)
    layer2 = FCNN(hidden_layers,hidden_layers,tf_log,d_tf_log,random_seed)
    layer3 = FCNN(hidden_layers,hidden_layers,tf_log,d_tf_log,random_seed)
    layer4 = FCNN(hidden_layers,20,tf_log,d_tf_log,random_seed)

    l1w,l2w,l3w,l4w = layer1.getw(),layer2.getw(),layer3.getw(),layer4.getw()

    # 4. Make the graph
    x = tf.placeholder(shape=[None,12],dtype=tf.float32)
    y = tf.placeholder(shape=[None,20],dtype=tf.float32)
    iter_variable_dil = tf.placeholder(tf.float32, shape=())
    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

    l1 = layer1.feedforward(x)
    l2 = layer2.feedforward(l1)
    l3 = layer3.feedforward(l2)
    l4 = layer4.feedforward(l3)
    final_soft = tf_softmax(l4)

    cost = tf.reduce_sum(-1.0 * (y* tf.log(final_soft) + (1-y)*tf.log(1-final_soft)))
    correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # --- Auto Train ----
    auto_train= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=[l1w,l2w,l3w,l4w])

    avg_over_time =[]
    cost_over_time =[]

    # 5. Train
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        avg_accuracy  = 0
        avg_cost = 0

        for iter in range(num_epoch):
            
            for batch_index in range(0,len(protein_data),batch_size):
                
                current_batch = dna_data[batch_index:batch_index+batch_size,:]
                current_batch_label = protein_data[batch_index:batch_index+batch_size,:]

                sess_results = sess.run([accuracy,cost,auto_train,correct_prediction],feed_dict={x:current_batch,y:current_batch_label})
                
                print("Current Iter: ", iter, " Current acuuracy: ", sess_results[0], " current cost: ",sess_results[1], end='\r')
                avg_accuracy = avg_accuracy + sess_results[0]
                avg_cost     = avg_cost + sess_results[1]

            if iter %print_size==0:
                print('\n')
                print("Avg Accuracy for iter: ", iter, " acc: ", avg_accuracy/(len(protein_data)/batch_size))

            avg_over_time.append(avg_accuracy/(len(protein_data)/batch_size))
            cost_over_time.append(avg_cost/(len(protein_data)/batch_size))
            avg_cost = 0
            avg_accuracy = 0
    return avg_over_time,cost_over_time


def man_diff(compare_number=None,random_seed=None):
    
    tf.reset_default_graph()
    print('\n\n========= Starting Comparison : ',compare_number, ' ==== Manual ========')
    
    # 3.Make the Layers
    layer1 = FCNN(12,hidden_layers,tf_log,d_tf_log,random_seed)
    layer2 = FCNN(hidden_layers,hidden_layers,tf_log,d_tf_log,random_seed)
    layer3 = FCNN(hidden_layers,hidden_layers,tf_log,d_tf_log,random_seed)
    layer4 = FCNN(hidden_layers,20,tf_log,d_tf_log,random_seed)

    # 4. Make the graph
    x = tf.placeholder(shape=[None,12],dtype=tf.float32)
    y = tf.placeholder(shape=[None,20],dtype=tf.float32)
    iter_variable_dil = tf.placeholder(tf.float32, shape=())
    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

    l1 = layer1.feedforward(x)
    l2 = layer2.feedforward(l1)
    l3 = layer3.feedforward(l2)
    l4 = layer4.feedforward(l3)
    final_soft = tf_softmax(l4)

    cost = tf.reduce_sum(-1.0 * (y* tf.log(final_soft) + (1-y)*tf.log(1-final_soft)))
    correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    l4g,l4update = layer4.backprop(final_soft - y)
    l3g,l3update = layer3.backprop(l4g)
    l2g,l2update = layer2.backprop(l3g+decay_propotoin_rate*l4g)
    l1g,l1update = layer1.backprop(l2g+decay_propotoin_rate*(l4g+l3g))
    weight_updates = l4update + l3update + l2update + l1update

    avg_over_time =[]
    cost_over_time =[]
    

    # 5. Train
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        avg_accuracy  = 0
        avg_cost = 0

        for iter in range(num_epoch):
            
            for batch_index in range(0,len(protein_data),batch_size):
                
                current_batch = dna_data[batch_index:batch_index+batch_size,:]
                current_batch_label = protein_data[batch_index:batch_index+batch_size,:]

                sess_results = sess.run([accuracy,cost,weight_updates,correct_prediction],feed_dict={x:current_batch, y:current_batch_label, iter_variable_dil:iter})
                
                print("Current Iter: ", iter, " Current acuuracy: ", sess_results[0], " current cost: ",sess_results[1], end='\r')
                avg_accuracy = avg_accuracy + sess_results[0]
                avg_cost     = avg_cost + sess_results[1]

            if iter %print_size==0:
                print('\n')
                print("Avg Accuracy for iter: ", iter, " acc: ", avg_accuracy/(len(protein_data)/batch_size))

            avg_over_time.append(avg_accuracy/(len(protein_data)/batch_size))
            cost_over_time.append(avg_cost/(len(protein_data)/batch_size))
            
            avg_cost = 0
            avg_accuracy = 0
    return avg_over_time,cost_over_time

if __name__=='__main__':
    
    man_avg_win_cost = 0
    man_cost_win_cost = 0
    auto_avg_win_cost = 0
    auto_cost_win_cost = 0

    for numer_compare in range(compare_range):
        random_seed_give = np.random.randint(999)
        man_avg,man_cost    = man_diff(numer_compare,random_seed_give)
        auto_avg,auto_cost = auto_diff(numer_compare,random_seed_give)


        plt.figure()
        plt.plot(range(len(man_avg)),man_avg,color='red')
        plt.plot(range(len(man_avg)),auto_avg,color='blue')
        plt.savefig(save_location+str(numer_compare)+'_avg_accuracy_graph.png')
        plt.figure()
        plt.plot(range(len(man_cost)),man_cost,color='red')
        plt.plot(range(len(man_cost)),auto_cost,color='blue')
        plt.savefig(save_location+str(numer_compare)+'_avg_cost_graph.png')

        man_last = man_avg[-1]
        auto_last = auto_avg[-1]
        man_cost_last = man_cost[-1]
        auto_cost_last = auto_cost[-1]

        avg_list = [man_last,auto_last]
        cost_list = [man_cost_last,auto_cost_last]
        
        won_avg =  avg_list.index(max(avg_list))
        won_cost = cost_list.index(min(cost_list))

        # print(man_avg)
        # print(man_cost)
    
        # print(auto_avg)
        # print(auto_cost)    
    
        # print(man_last)
        # print(auto_last)    
    
        # print(man_cost_last)
        # print(auto_cost_last)    

        # print(avg_list)    
        # print(cost_list)    

        # print(won_avg)    
        # print(won_cost)    

        plt.figure()
        plt.text(0, 0.8,"Max Avg Case: "+ str(won_avg) + " With value: "+ str(avg_list[won_avg]))
        plt.text(0, 0.6,"All Avg List " + str(avg_list))
        
        plt.text(0, 0.4,"Min Cost Case: "+ str(won_cost)+ " With value: "+ str(cost_list[won_cost]))
        plt.text(0, 0.2,"All Cost List " + str(cost_list))
        
        plt.savefig(save_location+str(numer_compare)+'final.png')

        if won_avg == 0:
            man_avg_win_cost= man_avg_win_cost + 1
        else :
            auto_avg_win_cost = auto_avg_win_cost + 1
            
        if won_cost == 0:
            man_cost_win_cost= man_cost_win_cost + 1
        else :
            auto_cost_win_cost = auto_cost_win_cost + 1

    plt.figure()
    plt.text(0, 0.8,"Man Avg Win Count: " + str(man_avg_win_cost))
    plt.text(0, 0.6,"Man cost Win Count: " + str(man_cost_win_cost))
    plt.text(0, 0.4,"Auto Avg Win Count: " + str(auto_avg_win_cost))
    plt.text(0, 0.2,"Auto cost Win Count: " + str(auto_cost_win_cost))
    plt.savefig(save_location+'z_final.png')
    
                

# -- end code --