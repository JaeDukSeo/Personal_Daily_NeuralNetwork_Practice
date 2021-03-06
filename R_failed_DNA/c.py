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

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLU(x):  return tf.cast(tf.greater(x, 0),dtype=tf.float32)    

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
num_training_data = 300
length_of_dna = 1
print_interval = 3

learning_rate_h = 0.000000001
learning_rate_x = 0.00000001
num_epoch = 300
l1_hid_num = 4
l2_hid_num = 1

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
dna_to_num = (dna_to_num-dna_to_num.min())/(dna_to_num.max()-dna_to_num.min())
protein_to_num = protein_data.view(np.uint8)
protein_to_num = (protein_to_num-protein_to_num.min())/(protein_to_num.max()-protein_to_num.min())

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
l1  = simpleRNN(tf_ReLU,d_tf_ReLU,length_of_dna*3+1,1,l1_hid_num)
l2  = simpleRNN(tf_log,d_tf_log,3+1,l1_hid_num,1)

# 1.8 Make the graph
x = tf.placeholder(shape=[1,length_of_dna*3],dtype=tf.float32)
y = tf.placeholder(shape=[1,length_of_dna],dtype=tf.float32)
output_list = tf.Variable(tf.zeros(length_of_dna))
output_assign = []
grad_update = []
timestamp = tf.constant(0)

layer1,l1_hidden = l1.feed_forward(tf.expand_dims(tf.expand_dims(x[0,timestamp+0],axis=0),axis=1),  time_stamp=timestamp)
layer2,l2_hidden = l1.feed_forward(tf.expand_dims(tf.expand_dims(x[0,timestamp+1],axis=0),axis=1),time_stamp=timestamp+1)
layer3,l3_hidden = l1.feed_forward(tf.expand_dims(tf.expand_dims(x[0,timestamp+2],axis=0),axis=1),time_stamp=timestamp+2)
layer21,l21_hidden = l2.feed_forward(layer3,time_stamp=timestamp)
output_assign.append([l1_hidden,l2_hidden,l3_hidden,l21_hidden])
output_assign.append(tf.assign(output_list[timestamp] ,tf.squeeze(layer21)) )

# layer4,l4_hidden = l1.feed_forward(tf.expand_dims(tf.expand_dims(x[0,timestamp+3],axis=0),axis=1),time_stamp=timestamp+3)
# layer5,l5_hidden = l1.feed_forward(tf.expand_dims(tf.expand_dims(x[0,timestamp+4],axis=0),axis=1),time_stamp=timestamp+4)
# layer6,l6_hidden = l1.feed_forward(tf.expand_dims(tf.expand_dims(x[0,timestamp+5],axis=0),axis=1),time_stamp=timestamp+5)
# layer22,l22_hidden = l2.feed_forward(layer6,time_stamp=timestamp+1)
# output_assign.append([l4_hidden,l5_hidden,l6_hidden,l22_hidden])
# output_assign.append(tf.assign(output_list[timestamp+1] ,tf.squeeze(layer22)) )

# layer7,l7_hidden = l1.feed_forward(tf.expand_dims(tf.expand_dims(x[0,timestamp+6],axis=0),axis=1),time_stamp=timestamp+6)
# layer8,l8_hidden = l1.feed_forward(tf.expand_dims(tf.expand_dims(x[0,timestamp+7],axis=0),axis=1),time_stamp=timestamp+7)
# layer9,l9_hidden = l1.feed_forward(tf.expand_dims(tf.expand_dims(x[0,timestamp+8],axis=0),axis=1),time_stamp=timestamp+8)
# layer23,l23_hidden = l2.feed_forward(layer9,time_stamp=timestamp+2)
# output_assign.append([l7_hidden,l8_hidden,l9_hidden,l23_hidden])
# output_assign.append(tf.assign(output_list[timestamp+2] ,tf.squeeze(layer23)) )

cost1 = tf.square(output_list[timestamp]-  y[0,timestamp])
# cost2 = tf.square(output_list[timestamp+1]-y[0,timestamp+1])
# cost3 = tf.square(output_list[timestamp+2]-y[0,timestamp+2])
# total_cost = cost1 + cost2 + cost3 
total_cost = cost1 

# grad23,grad23_update = l2.backpropagation(tf.expand_dims(output_list[timestamp+2]-y[0,timestamp+2],axis=0),timestamp+2)
# grad9,grad9_update = l1.backpropagation(grad23,timestamp+8)
# grad8,grad8_update = l1.backpropagation(grad9,timestamp+7)
# grad7,grad7_update = l1.backpropagation(grad8,timestamp+6)

# grad22,grad22_update = l2.backpropagation(tf.expand_dims(output_list[timestamp+1]-y[0,timestamp+1],axis=0),timestamp+1)
# grad6,grad6_update = l1.backpropagation(grad7+grad22,timestamp+5)
# grad5,grad5_update = l1.backpropagation(cost3,timestamp+4)
# grad4,grad4_update = l1.backpropagation(grad6,timestamp+3)

grad21,grad21_update = l2.backpropagation(tf.expand_dims(output_list[timestamp]-  y[0,timestamp],axis=0),timestamp)
grad3,grad3_update = l1.backpropagation(grad21,timestamp+2)
grad2,grad2_update = l1.backpropagation(grad3,timestamp+1)
grad1,grad1_update = l1.backpropagation(grad2,timestamp+0)
grad_update.append([grad21_update,  grad3_update,grad2_update,grad1_update])
# grad_update.append([grad23_update,grad9_update,grad8_update,grad7_update,
#                     grad22_update,  grad6_update,grad5_update,grad4_update,
#                     grad21_update,  grad3_update,grad2_update,grad1_update])


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

            # sess_result = sess.run([layer1,l1_hidden],feed_dict={x:current_dna_data,y:current_proten_data})
            # print(sess_result[0].shape)
            # sys.exit()

            sess_result = sess.run([total_cost,output_assign,grad_update],feed_dict={x:current_dna_data,y:current_proten_data})
            print("Current Iter: ", iter,' Current train index: ',current_train_index, ' Current Total Cost: ',sess_result[0],end='\r')
            total_cost_current= total_cost_current+sess_result[0]

        if iter % print_interval == 0 :
            print("\n===== Testing Middle ======")

            random_testing_index = np.random.randint(dna_to_num.shape[0])
            testing = np.expand_dims(dna_to_num[random_testing_index,:],axis=0)
            gt      = protein_to_num[random_testing_index,:]
            # sess_result = sess.run([layer21,layer22,layer23,output_assign],feed_dict={x:testing})
            sess_result = sess.run([layer21,output_assign],feed_dict={x:testing})
            
            print("Predicted: ",sess_result[0])
            # print("Predicted: ",sess_result[0],sess_result[1],sess_result[2])
            print("GT : " ,gt)
            print("Total cost now: ", total_cost_current)
            print("===== Testing Middle ======\n")

        cost_over_time.append(total_cost_current)
        total_cost_current = 0


            # sys.exit()
# # a. Training Epoch
# for iter in range(num_epoch):
    
#     # b. For Each DNA lines
#     for current_train_index in range(num_training_data):
        
#         current_dna_data = dna_to_num[current_train_index,:]
#         current_proten_data = protein_to_num[current_train_index,:]
#         output_list = np.zeros((length_of_dna*3+1))

#         # c. Get the line of the protein
#         for data_index in range(len(current_proten_data)):

#             # d. Loop via the DNA as well ( Feed Forward Here )
#             for dna_data_index in range(data_index*3,data_index*3+3):
#                 # print(current_dna_data[dna_data_index],end=' ')
#                 # print(dna_data_index,end=' ')
#                 layer_ts_1 = l1.feed_forward(float32(current_dna_data[dna_data_index]),time_stamp=dna_data_index)
#             print('\n')
#             print(current_proten_data[data_index],end='\n')
#             print('======================')

#         print('---------------------------------')
        
                
            
            







# Convert Back To char
ss = protein_to_num.view('c')




# ----- end code ----