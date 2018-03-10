import tensorflow as tf
import random
import sys
import numpy as np
from numpy import float32
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
np.random.seed(678)
tf.set_random_seed(768)

# Activation Functions - however there was no indication in the original paper
def tf_log(x): 
    return tf.sigmoid(x)

def d_tf_log(x): 
    return tf.multiply(tf_log(x),tf.subtract(1.0,tf_log(x))) 

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

learning_rate = 0.0005

beta_1 ,beta_2= 0.9, 0.999
adam_e = 0.00000001

num_epoch = 800
print_size = 50

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
    
    def __init__(self,input=None,output=None,act=None,d_act=None):

        self.w = tf.Variable(tf.random_normal([input,output]))

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

# 3.Make the Layers
layer1 = FCNN(12,48,tf_log,d_tf_log)
layer2 = FCNN(48,75,tf_log,d_tf_log)
layer3 = FCNN(75,140,tf_log,d_tf_log)
layer4 = FCNN(140,20,tf_log,d_tf_log)

l1w,l2w,l3w,l4w = layer1.getw(),layer2.getw(),layer3.getw(),layer4.getw()

# 4. Make the graph
x = tf.placeholder(shape=[None,12],dtype=tf.float32)
y = tf.placeholder(shape=[None,20],dtype=tf.float32)
decay_rate = 

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

l4g,l4update = layer4.backprop(final_soft - y)
l3g,l3update = layer3.backprop(l4g)
l2g,l2update = layer2.backprop(l3g)
l1g,l1update = layer1.backprop(l2g)
weight_updates = l4update + l3update + l2update + l1update

# 5. Train
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        for batch_index in range(0,len(protein_data),100):
            
            current_batch = dna_data[batch_index:batch_index+100,:]
            current_batch_label = protein_data[batch_index:batch_index+100,:]

            # sess_results = sess.run([accuracy,auto_train],feed_dict={x:current_batch,y:current_batch_label})
            sess_results = sess.run([accuracy,cost,weight_updates,correct_prediction],feed_dict={x:current_batch,y:current_batch_label})
            
            print("Current Iter: ", iter, " Current acuuracy: ", sess_results[0], " current cost: ",sess_results[1], end='\r')

        if iter %print_size==0:
            print('\n')
            





# -- end code --