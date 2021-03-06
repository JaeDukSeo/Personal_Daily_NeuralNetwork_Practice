import tensorflow as tf
import random
import sys
import numpy as np
from numpy import float32

np.random.seed(678)
tf.set_random_seed(768)


def tf_log(x):
    return tf.sigmoid(x)
def d_tf_log(x):
    return tf_log(x) * (1.0 - tf_log(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1-tf.square(tf_tanh(x) ) 

# -1. DNA codon table
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
Terminal_Sig = ['TAA','TAG','TGA']

# 0. Generate Random Data of dna to convert to Protein
num_training_data = 3
length_of_dna = 9

dna_data = np.array([])
protein_data = np.array([])

for training_index in range(num_training_data):
    
    current_data = np.array([])
    protein_sequence = np.array([])
    
    for _ in range(length_of_dna):
        dna,protein = random.choice(list(protein_table.items()))
        for d in dna:
            current_data = np.append(current_data,d)
        protein_sequence = np.append(protein_sequence,[protein])

    if training_index == 0 :
        dna_data = np.expand_dims(current_data,axis=0)
        protein_data = np.expand_dims(protein_sequence,axis=0)
    else:
        dna_data = np.vstack((dna_data,np.expand_dims(current_data,axis=0)))
        protein_data = np.vstack((protein_data,np.expand_dims(protein_sequence,axis=0)))

# 0.15 Conver to numeric value
dna_data = dna_data.astype('c')
protein_data = protein_data.astype('c')

print('---- the shape of training data -----')
print(dna_data.shape)
print(protein_data.shape)
print('---- the shape of training data -----')


# 0.5 Convert letter to digit
dna_to_num = dna_data.view(np.uint8)
protein_to_num = protein_data.view(np.uint8)

# 1. Create the Model
class simpleRNN():

    def __init__(self,activation,d_activation,hidden_lenght):

        self.w_x = tf.Variable(tf.random_normal([]))
        self.w_h = tf.Variable(tf.random_normal([]))
        self.h = tf.zeros([hidden_lenght])
        self.act = activation
        self.d_act = d_activation

    def feed_forward(self,input=None,time_stamp=None):

        self.input = input
        self.layer = tf.multiply(input,self.w_x)  + tf.multiply(self.h[time_stamp],self.w_h)  

        return self.act(self.input)

# 1.5 Make the objects and graphs
l1 = simpleRNN(tf_log,d_tf_log,length_of_dna*3+1)

x = tf.placeholder()



# 2. Declare Hyper Parameteres
num_epoch = 1

# 3. Start the training
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # a. Training Epoch
    for iter in range(num_epoch):
        
        # b. For Each DNA lines
        for current_train_index in range(num_training_data):
            
            current_dna_data = dna_to_num[current_train_index,:]
            current_proten_data = protein_to_num[current_train_index,:]
            
            # c. Get the line of the protein
            for data_index in range(len(current_proten_data)):

                # d. Loop via the DNA as well ( Feed Forward Here )
                for dna_data_index in range(data_index*3,data_index*3+3):
                    # print(current_dna_data[dna_data_index],end=' ')
                    # print(dna_data_index,end=' ')
                    layer_ts_1 = l1.feed_forward(float32(current_dna_data[dna_data_index]),time_stamp=dna_data_index)
                print('\n')
                print(current_proten_data[data_index],end='\n')
                print('======================')

            print('---------------------------------')
        
                
            
            







# Convert Back To char
ss = protein_to_num.view('c')




# ----- end code ----