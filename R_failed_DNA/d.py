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
protein_table = {"TTT" : 1.0, "CTT" : 2.0, "ATT" : 12.0, "GTT" : 18.0,
           "TTC" : 1.0, "CTC" : 2.0, "ATC" : 12.0, "GTC" : 18.0,
           "TTA" : 2.0, "CTA" : 2.0, "ATA" : 12.0, "GTA" : 18.0,
           "TTG" : 2.0, "CTG" : 2.0, "ATG" : 13.0, "GTG" : 18.0,
           "TCT" : 3.0, "CCT" : 9.0, "ACT" : 14.0, "GCT" : 19.0,
           "TCC" : 3.0, "CCC" : 9.0, "ACC" : 14.0, "GCC" : 19.0,
           "TCA" : 3.0, "CCA" : 9.0, "ACA" : 14.0, "GCA" : 19.0,
           "TCG" : 3.0, "CCG" : 9.0, "ACG" : 14.0, "GCG" : 19.0,
           "TAT" : 4.0, "CAT" : 10.0, "AAT" : 15.0, "GAT" : 20.0,
           "TAC" : 4.0, "CAC" : 10.0, "AAC" : 15.0, "GAC" : 20.0,
            "CAA" : 5.0, "AAA" : 11.0, "GAA" : 16.0,
            "CAG" : 5.0, "AAG" : 11.0, "GAG" : 16.0,
           "TGT" : 6.0, "CGT" : 7.0, "AGT" : 3.0, "GGT" : 17.0,
           "TGC" : 6.0, "CGC" : 7.0, "AGC" : 3.0, "GGC" : 17.0,
            "CGA" : 7.0, "AGA" : 7.0, "GGA" : 17.0,
           "TGG" : 8.0, "CGG" : 7.0, "AGG" : 7.0, "GGG" : 17.0 
           }

Terminal_Sig = ['TAA','TAG','TGA']

# 0. Generate Random Data of dna to convert to Protein
num_training_data = 300
length_of_dna = 1
print_interval = 3

learning_rate_x = 0.00001
num_epoch = 300

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
# protein_data = protein_data.astype('c')

print('---- the shape of training data -----')
print(dna_data.shape)
print(protein_data.shape)
print('---- the shape of training data -----')

# 0.5 Convert letter to digit
dna_to_num = dna_data.view(np.uint8)
dna_to_num = (dna_to_num-dna_to_num.min())/(dna_to_num.max()-dna_to_num.min())
protein_to_num = protein_data

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
l1  = fnnlayer(tf_log,d_tf_log,3,5)
l2  = fnnlayer(tf_tanh,d_tf_tanh,5,8)
l3  = fnnlayer(tf_log,d_tf_log,8,10)
l4  = fnnlayer(tf_log,d_tf_log,10,1)

l1w,l2w,l3w,l4w = l1.getw(),l2.getw(),l3.getw(),l4.getw()

# 1.8 Make the graph
x = tf.placeholder(shape=[1,length_of_dna*3],dtype=tf.float32)
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
            
            current_dna_data = np.expand_dims(dna_to_num[current_train_index,:],axis=0)
            current_proten_data = np.expand_dims(protein_to_num[current_train_index,:],axis=0)

            sess_result = sess.run([total_cost,auto_grad],feed_dict={x:current_dna_data,y:current_proten_data})
            print("Current Iter: ", iter,' Current train index: ',current_train_index, ' Current Total Cost: ',sess_result[0],end='\r')
            total_cost_current= total_cost_current+sess_result[0]

        if iter % print_interval == 0 :
            print("\n===== Testing Middle ======")
            random_testing_index = np.random.randint(dna_to_num.shape[0])
            testing = np.expand_dims(dna_to_num[random_testing_index,:],axis=0)
            gt      = protein_to_num[random_testing_index,:]
            sess_result = sess.run([layer4],feed_dict={x:testing})
            print("Predicted: ",sess_result[0])
            print("GT : " ,gt)
            print("Total cost now: ", total_cost_current)
            print("===== Testing Middle ======\n")

        cost_over_time.append(total_cost_current)
        total_cost_current = 0





# ----- end code ----