import tensorflow as tf
import random
import sys
import numpy as np
from numpy import float32
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
np.random.seed(678)
tf.set_random_seed(768)


def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x):return tf_log(x) * (1.0 - tf_log(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1-tf.square(tf_tanh(x) ) 

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLU(x):  return tf.cast(tf.greater(x, 0),dtype=tf.float32)    


def tf_softmax(x=None): return tf.nn.softmax(x)

# -1. DNA codon table
# protein_table = {"TTT" : -0.9, "CTT" : -0.8, "ATT" : 0.2, "GTT" : 0.8,
#            "TTC" : -0.9, "CTC" : -0.8, "ATC" : 0.2, "GTC" : 0.8,
#            "TTA" : -0.8, "CTA" : -0.8, "ATA" : 0.2, "GTA" : 0.8,
#            "TTG" : -0.8, "CTG" : -0.8, "ATG" : 0.3, "GTG" : 0.8,
#            "TCT" : -0.7, "CCT" : -0.1, "ACT" : 0.4, "GCT" : 0.9,
#            "TCC" : -0.7, "CCC" : -0.1, "ACC" : 0.4, "GCC" : 0.9,
#            "TCA" : -0.7, "CCA" : -0.1, "ACA" : 0.4, "GCA" : 0.9,
#            "TCG" : -0.7, "CCG" : -0.1, "ACG" : 0.4, "GCG" : 0.9,
#            "TAT" : -0.6, "CAT" : 0.0, "AAT" : 0.5, "GAT" : 1.0,
#            "TAC" : -0.6, "CAC" : 0.0, "AAC" : 0.5, "GAC" : 1.0,
#             "CAA" : -0.5, "AAA" : 0.1, "GAA" : 0.6,
#             "CAG" : -0.5, "AAG" : 0.1, "GAG" : 0.6,
#            "TGT" : -0.4, "CGT" : -0.3, "AGT" : -0.7, "GGT" : 0.7,
#            "TGC" : -0.4, "CGC" : -0.3, "AGC" : -0.7, "GGC" : 0.7,
#             "CGA" : -0.3, "AGA" : -0.3, "GGA" : 0.7,
#            "TGG" : -0.2, "CGG" : -0.3, "AGG" : -0.3, "GGG" : 0.7 
#            }


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
num_training_data = 800
length_of_dna = 1
print_interval = 3

learning_rate_h = 0.0000001
learning_rate_x = 0.00001
num_epoch = 100

l1_hid_num = 25
l2_hid_num = 30

# 1. Array to Contain all of the data 
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

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(protein_data)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
protein_data = onehot_encoder.fit_transform(integer_encoded)


# 1. Create the Model
class simpleRNN():

    def __init__(self,activation,d_activation,hidden_lengths,input_dim=None,hidden_dim=None):

        self.w_x = tf.Variable(tf.truncated_normal([input_dim,hidden_dim]))
        self.w_h = tf.Variable(tf.truncated_normal([hidden_dim,hidden_dim]))

        self.h   = tf.Variable(tf.zeros([hidden_dim,hidden_lengths]))
        self.h_I = tf.Variable(tf.zeros([hidden_dim,hidden_lengths]))
        
        self.input = tf.Variable(tf.zeros([input_dim,hidden_lengths]))

        self.act = activation
        self.d_act = d_activation

    def feed_forward(self,input=None,time_stamp=None):
        
        self.layer = tf.matmul(input,self.w_x)  + tf.matmul(tf.expand_dims(self.h[:,time_stamp],axis=0),self.w_h)  
        self.layerA = self.act(self.layer)

        assigns = []
        assigns.append(tf.assign(self.input[:,time_stamp],tf.squeeze(input,axis=0)))
        assigns.append(tf.assign(self.h_I[:,time_stamp+1],tf.squeeze(self.layer,axis=0)))
        assigns.append(tf.assign(self.h[:,time_stamp+1],tf.squeeze(self.layerA,axis=0)))

        return self.layerA,assigns

    def getw(self): return [self.w_x,self.w_h]

    def backpropagation(self,gradient=None,time_stamp=None):
        
        grad_part_1 = gradient
        grad_part_2 = tf.expand_dims(self.d_act(self.h_I[:,time_stamp]),axis=0)
        grad_part_h = tf.expand_dims(self.h[:,time_stamp-1],axis=0)
        grad_part_x = tf.expand_dims(self.input[:,time_stamp],axis=0)

        grad_h_mid = tf.multiply(grad_part_1,grad_part_2)
        grad_h = tf.matmul(tf.transpose(grad_part_h),grad_h_mid)

        grad_x_mid = tf.multiply(grad_part_1,grad_part_2)
        grad_x = tf.matmul(tf.transpose(grad_part_x),grad_x_mid)
        
        grad_pass = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(grad_part_h))
        
        w_update = [tf.assign(self.w_x,tf.subtract(self.w_x,learning_rate_x*grad_x) )]
        w_update.append(tf.assign(self.w_h,tf.subtract(self.w_h,learning_rate_h*grad_h) ))
        
        return grad_pass,w_update

# 1.5 Make the objects and graphs
l1  = simpleRNN(tf_log,d_tf_log,length_of_dna*3+1,4,l1_hid_num)
l2  = simpleRNN(tf_tanh,d_tf_tanh,length_of_dna*3+1,l1_hid_num,l2_hid_num)
l3  = simpleRNN(tf_tanh,d_tf_tanh,length_of_dna*3+1,l2_hid_num,20)

l1w,l2w = l1.getw(),l2.getw()

# 1.8 Make the graph
x = tf.placeholder(shape=[4,length_of_dna*3],dtype=tf.float32)
y = tf.placeholder(shape=[1,20],dtype=tf.float32)
output_assign = []
grad_update = []
timestamp = tf.constant(0)

layer1,l1_hidden =   l1.feed_forward(tf.expand_dims(x[:,timestamp+0],axis=0),  time_stamp=timestamp+0)
layer21,l21_hidden = l2.feed_forward(layer1,time_stamp=timestamp+0)
layer31,l31_hidden = l3.feed_forward(layer21,time_stamp=timestamp+0)

layer2,l2_hidden =   l1.feed_forward(tf.expand_dims(x[:,timestamp+1],axis=0),  time_stamp=timestamp+1)
layer22,l22_hidden = l2.feed_forward(layer2,time_stamp=timestamp+1)
layer32,l32_hidden = l3.feed_forward(layer22,time_stamp=timestamp+1)

layer3,l3_hidden =   l1.feed_forward(tf.expand_dims(x[:,timestamp+2],axis=0),  time_stamp=timestamp+2)
layer23,l23_hidden = l2.feed_forward(layer3,time_stamp=timestamp+2)
layer33,l33_hidden = l3.feed_forward(layer23,time_stamp=timestamp+2)

output_assign.append([l1_hidden,l2_hidden,l3_hidden,
                      l21_hidden,l22_hidden,l23_hidden,
                      l31_hidden,l32_hidden,l33_hidden])


final_softmax = tf_softmax(layer33)
total_cost = tf.reduce_sum( -1 * ( y*tf.log(final_softmax) + (1-y) * tf.log(1-final_softmax ) ) )

correct_prediction = tf.equal(tf.argmax(layer33, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


grad33,grad33_update = l3.backpropagation(final_softmax-  y,timestamp+2)
grad32,grad32_update = l3.backpropagation(grad33,timestamp+1)
grad31,grad31_update = l3.backpropagation(grad32,timestamp)

grad23,grad23_update = l2.backpropagation(grad33,timestamp+2)
grad22,grad22_update = l2.backpropagation(grad23+grad32,timestamp+1)
grad21,grad21_update = l2.backpropagation(grad22+grad31,timestamp)

grad3,grad3_update = l1.backpropagation(grad23,timestamp+2)
grad2,grad2_update = l1.backpropagation(grad3+grad22,timestamp+1)
grad1,grad1_update = l1.backpropagation(grad2+grad21,timestamp+0)

grad_update.append([grad33_update,grad32_update,grad31_update,  
                    grad23_update,grad22_update,grad21_update,  
                    grad3_update,grad2_update,grad1_update])

# 3. Start the training
with tf.Session() as sess:

    total_cost_current = 0
    cost_over_time = []
    sess.run(tf.global_variables_initializer())

    # a. Training Epoch
    for iter in range(num_epoch):
        
        # b. For Each DNA lines
        for current_train_index in range(num_training_data):
            
            current_dna_data = np.reshape(dna_data[current_train_index,:],(4,3))
            current_proten_data = np.expand_dims(protein_data[current_train_index,:],axis=0)
            sess_result = sess.run([total_cost,output_assign,grad_update],feed_dict={x:current_dna_data,y:current_proten_data})
            # sess_result = sess.run([total_cost,auto_train],feed_dict={x:current_dna_data,y:current_proten_data})
            print("Current Iter: ", iter,' Current train index: ',current_train_index, ' Current Total Cost: ',sess_result[0],end='\r')
            total_cost_current= total_cost_current+sess_result[0]

        if iter % print_interval == 0 :

            print("\n===== Testing Middle ======")
            for _ in range(3):
                random_testing_index = np.random.randint(dna_data.shape[0])
                testing = np.reshape(dna_data[random_testing_index,:],(4,3))
                gt      = protein_data[random_testing_index,:]
                sess_result = sess.run([final_softmax,output_assign],feed_dict={x:testing})
                print("Predicted: ",np.argmax(sess_result[0], axis=1))
                print("Predicted: ",sess_result[0])
                print("GT : " ,gt)
            print("Total cost now: ", total_cost_current)
            print("===== Testing Middle ======\n")
        cost_over_time.append(total_cost_current)
        total_cost_current = 0












# ----- end code ----