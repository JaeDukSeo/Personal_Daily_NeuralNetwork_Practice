import tensorflow as tf
import random
import sys
import numpy as np
from numpy import float32

np.random.seed(678)
tf.set_random_seed(768)


def tf_log(x):
    return 20.0*tf.sigmoid(x)
def d_tf_log(x):
    return 20.0*(tf_log(x) * (1.0 - tf_log(x)))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1-tf.square(tf_tanh(x) ) 

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLU(x):  return tf.cast(tf.greater(x, 0),dtype=tf.float32)    

# -1. DNA codon table
protein_table = {"TTT" : -0.9, "CTT" : -0.8, "ATT" : 0.2, "GTT" : 0.8,
           "TTC" : -0.9, "CTC" : -0.8, "ATC" : 0.2, "GTC" : 0.8,
           "TTA" : -0.8, "CTA" : -0.8, "ATA" : 0.2, "GTA" : 0.8,
           "TTG" : -0.8, "CTG" : -0.8, "ATG" : 0.3, "GTG" : 0.8,
           "TCT" : -0.7, "CCT" : -0.1, "ACT" : 0.4, "GCT" : 0.9,
           "TCC" : -0.7, "CCC" : -0.1, "ACC" : 0.4, "GCC" : 0.9,
           "TCA" : -0.7, "CCA" : -0.1, "ACA" : 0.4, "GCA" : 0.9,
           "TCG" : -0.7, "CCG" : -0.1, "ACG" : 0.4, "GCG" : 0.9,
           "TAT" : -0.6, "CAT" : 0.0, "AAT" : 0.5, "GAT" : 1.0,
           "TAC" : -0.6, "CAC" : 0.0, "AAC" : 0.5, "GAC" : 1.0,
            "CAA" : -0.5, "AAA" : 0.1, "GAA" : 0.6,
            "CAG" : -0.5, "AAG" : 0.1, "GAG" : 0.6,
           "TGT" : -0.4, "CGT" : -0.3, "AGT" : -0.7, "GGT" : 0.7,
           "TGC" : -0.4, "CGC" : -0.3, "AGC" : -0.7, "GGC" : 0.7,
            "CGA" : -0.3, "AGA" : -0.3, "GGA" : 0.7,
           "TGG" : -0.2, "CGG" : -0.3, "AGG" : -0.3, "GGG" : 0.7 
           }

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
num_training_data = 300
length_of_dna = 1
print_interval = 3

learning_rate_x = 0.000000001
num_epoch = 300

dna_data = np.array([])
protein_data = np.array([])

for training_index in range(num_training_data):
    
    current_data = np.array([])
    protein_sequence = np.array([])
    
    for _ in range(length_of_dna):
        dna,protein = random.choice(list(protein_table.items()))
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

print('---- the shape of training data -----')
print(dna_data.shape)
print(protein_data.shape)
print('---- the shape of training data -----')

sys.exit()

beta_1,beta_2 = 0.9,0.999
adam_e = 0.00000001

# 1. Create the Model
class fnnlayer():
    def __init__(self,activation,d_activation,input_dim=None,hidden_dim=None):
    
        self.w = tf.Variable(tf.random_normal([input_dim,hidden_dim]))
        self.b = tf.Variable(tf.random_normal([1,hidden_dim]))
        self.act = activation
        self.d_act = d_activation

        self.m,self.h = tf.Variable(tf.zeros([input_dim,hidden_dim])),tf.Variable(tf.zeros([input_dim,hidden_dim]))

    def getw(self): return self.w,self.b

    def feed_forward(self,input=None):
        
        self.input = input
        self.layer = tf.matmul(input,self.w) + self.b 
        self.layerA = self.act(self.layer)
        return self.layerA

    def backpropagation(self,gradient=None,time_stamp=None):
        
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input 

        grad_x_mid = tf.multiply(grad_part_1,grad_part_2)
        grad_x = tf.matmul(tf.transpose(grad_part_3),grad_x_mid)

        w_update = []
        w_update.append(tf.assign(self.m,self.m*beta_1+(1-beta_1)*grad_x))
        w_update.append(tf.assign(self.h,self.h*beta_2+(1-beta_2)*grad_x**2))

        m_hat = self.m/(1-beta_1)
        h_hat = self.h/(1-beta_2)
        
        grad_pass = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))
        
        w_update.append(tf.assign(self.w,tf.subtract(self.w,  learning_rate_x/(tf.sqrt(h_hat)+adam_e)*m_hat) ))
        
        return grad_pass,w_update

# 1.5 Make the objects and graphs
l1  = fnnlayer(tf_log,d_tf_log,12,15)
l2  = fnnlayer(tf_tanh,d_tf_tanh,15,28)
l3  = fnnlayer(tf_log,d_tf_log,28,40)
l4  = fnnlayer(tf_tanh,d_tf_tanh,40,1)

l1w,l2w,l3w,l4w = l1.getw(),l2.getw(),l3.getw(),l4.getw()

# 1.8 Make the graph
x = tf.placeholder(shape=[1,length_of_dna*12],dtype=tf.float32)
y = tf.placeholder(shape=[1,length_of_dna],dtype=tf.float32)
grad_update = []

layer1 = l1.feed_forward(x)
layer2 = l2.feed_forward(layer1)
layer3 = l3.feed_forward(layer2)
layer4 = l4.feed_forward(layer3)

total_cost = tf.reduce_sum(tf.square(layer4-y))
auto_grad = tf.train.AdamOptimizer(learning_rate=learning_rate_x).minimize(total_cost,var_list=[l1w,l2w,l3w,l4w ])

grad4,grad4_u=l4.backpropagation(layer4-y)
grad3,grad3_u=l3.backpropagation(grad4)
grad2,grad2_u=l2.backpropagation(grad3)
grad1,grad1_u=l1.backpropagation(grad2)
grad_update.append([grad1_u,grad2_u,grad3_u,grad4_u])

# 3. Start the training
with tf.Session() as sess:

    total_cost_current = 0
    cost_over_time = []
    sess.run(tf.global_variables_initializer())

    # a. Training Epoch
    for iter in range(num_epoch):
        
        # b. For Each DNA lines
        for current_train_index in range(num_training_data):
            
            current_dna_data = np.expand_dims(dna_data[current_train_index,:],axis=0)
            current_proten_data = np.expand_dims(protein_data[current_train_index,:],axis=0)

            sess_result = sess.run([total_cost,auto_grad],feed_dict={x:current_dna_data,y:current_proten_data})
            print("Current Iter: ", iter,' Current train index: ',current_train_index, ' Current Total Cost: ',sess_result[0],end='\r')
            total_cost_current= total_cost_current+sess_result[0]

        if iter % print_interval == 0 :
            print("\n===== Testing Middle ======")
            random_testing_index = np.random.randint(dna_data.shape[0])
            testing = np.expand_dims(dna_data[random_testing_index,:],axis=0)
            gt      = protein_data[random_testing_index,:]
            sess_result = sess.run([layer4],feed_dict={x:testing})
            print("Predicted: ",sess_result[0])
            print("GT : " ,gt)
            print("Total cost now: ", total_cost_current)
            print("===== Testing Middle ======\n")

        cost_over_time.append(total_cost_current)
        total_cost_current = 0





# ----- end code ----