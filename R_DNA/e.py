import tensorflow as tf
import random
import sys
import numpy as np
from numpy import float32
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
np.random.seed(678)
tf.set_random_seed(768)




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
num_training_data = 500
length_of_protein = 1
print_interval = 3

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
print(protein_data.shape)
print(dna_data.shape)

# 2. Make the class
class FCNN:
    
    def __init__(self,input=None,output=None,act=None,d_act=None):
        
        self.w = tf.Variable(tf.random_normal([input,output]))
        self.act = act
        self.d_act = d_act

    def feedforward(self,input=None):
        
        self.layer = tf.matmul(input,self.w)

layer1 = FCNN(12,100)


# -- end code --