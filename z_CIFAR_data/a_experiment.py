import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from read_10_data import get_std_mean_data,get_data

# activation functions here and there
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf_elu(x) + 1.0 

def tf_softmax(x): return tf.nn.softmax(x)


np.random.seed(678)
tf.set_random_seed(678)

# train_images,train_labels,test_images,test_labels = get_std_mean_data()
train_images,train_labels,test_images,test_labels = get_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# === Make Model ===
class CNN_Model_1():
    
    def __init__(self,kernel,in_c,out_c):
        self.w1 = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c],stddev=1e-8))
        self.w2 = tf.Variable(tf.truncated_normal([kernel,kernel,out_c,out_c],stddev=1e-8))
        self.w_rec = tf.Variable(tf.truncated_normal([1,1,in_c,out_c],stddev=1e-8))

    def feedforward(self,input,strides=1):
        self.layer = tf.nn.conv2d(input,self.w1,strides = [1,strides,strides,1],padding='SAME')
        self.layerA = tf_elu(self.layer)
        self.layer = tf.nn.conv2d(self.layerA,self.w2,strides = [1,1,1,1],padding='SAME')
        self.layerA = tf_elu(self.layer)
        self.layer_rec = tf.nn.conv2d(input,self.w_rec,strides = [1,strides,strides,1],padding='SAME')
        return self.layer_rec + self.layerA


class CNN_Model_2():
    def __init__(self,kernel,in_c,out_c):
        self.w1 = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c],stddev=1e-8))
        self.w2 = tf.Variable(tf.truncated_normal([kernel,kernel,out_c,out_c],stddev=1e-8))
        self.w3 = tf.Variable(tf.truncated_normal([kernel,kernel,out_c,out_c],stddev=1e-8))

    def feedforward(self,input,strides=1):
        self.layer = tf.nn.conv2d(input,self.w1,strides = [1,strides,strides,1],padding='SAME')
        self.layerA = tf_elu(self.layer)
        self.layer = tf.nn.conv2d(self.layerA,self.w2,strides = [1,1,1,1],padding='SAME')
        self.layerA = tf_elu(self.layer)
        self.layer = tf.nn.conv2d(self.layerA,self.w3,strides = [1,1,1,1],padding='SAME')
        self.layerA = tf_elu(self.layer)
        return input + self.layerA

# === hyper ====
batch_size = 100
print_size = 1
divide_size = 4
shuffle_size = 2

learning_rate_dynamic = 10
momentum_rate = 0.9

num_epoch = 100

# === Make Model ===
l1_1 = CNN_Model_1(3,3,16)
l1_2 = CNN_Model_2(3,16,16)

l2_1 = CNN_Model_1(5,16,64)
l2_2 = CNN_Model_2(3,64,64)

l3_1 = CNN_Model_1(5,64,256)
l3_2 = CNN_Model_2(3,256,256)

l4_1 = CNN_Model_1(5,256,10)
l4_2 = CNN_Model_2(3,10,10)

# === Make Graph ===
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
learning_rate = tf.placeholder(tf.float32)

w_connection1 = tf.Variable(tf.truncated_normal([1,1,3,16],stddev=1e-8))
w_connection2 = tf.Variable(tf.truncated_normal([1,1,3,64],stddev=1e-8))
w_connection3 = tf.Variable(tf.truncated_normal([1,1,3,256],stddev=1e-8))

layer1_1 = l1_1.feedforward(x,2)
layer1_2 = l1_2.feedforward(layer1_1)
# layer1_connection  = tf.nn.conv2d(x,w_connection1,strides=[1,2,2,1],padding='SAME')
# layer1_connected = tf.concat((layer1_2,layer1_connection),axis=3)

layer2_1 = l2_1.feedforward(layer1_2,2)
layer2_2 = l2_2.feedforward(layer2_1)
# layer2_connection  = tf.nn.conv2d(x,w_connection2,strides=[1,4,4,1],padding='SAME')
# layer2_connected = tf.concat((layer2_2,layer2_connection),axis=3)

layer3_1 = l3_1.feedforward(layer2_2,2)
layer3_2 = l3_2.feedforward(layer3_1)
# layer3_connection  = tf.nn.conv2d(x,w_connection3,strides=[1,8,8,1],padding='SAME')
# layer3_connected = tf.concat((layer3_2,layer3_connection),axis=3)

layer4_1 = l4_1.feedforward(layer3_2,4)
layer4_2 = l4_2.feedforward(layer4_1)


# --- final layer ----
final_soft = tf_softmax(tf.reshape(layer4_2,[batch_size,-1]))
cost = tf.reduce_sum(-1.0 * (y*tf.log(final_soft) + (1.0-y)*tf.log(1.0-final_soft)))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- auto train ---
# auto_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum_rate).minimize(cost)
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)










# === Start the Session ===
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.Session() as sess: 

  sess.run(tf.global_variables_initializer())

  train_total_cost,train_total_acc =0,0
  train_cost_overtime,train_acc_overtime = [],[]

  test_total_cost,test_total_acc = 0,0
  test_cost_overtime,test_acc_overtime = [],[]

  for iter in range(num_epoch):

        # Train Set
        for current_batch_index in range(0,int(len(train_images)/divide_size),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_labels[current_batch_index:current_batch_index+batch_size,:]

            if iter == 200: 
                learning_rate_dynamic = learning_rate_dynamic * 0.1

            sess_results =  sess.run([cost,accuracy,auto_train],feed_dict={x: current_batch, y: current_batch_label,learning_rate:learning_rate_dynamic})
            print("current iter:", iter,' Current batach : ',current_batch_index," current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
          current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
          current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
          sess_results =  sess.run([cost,accuracy],feed_dict={x: current_batch, y: current_batch_label})
          print("\t\t\tTest Image Current iter:", iter,' Current batach : ',current_batch_index, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
          test_total_cost = test_total_cost + sess_results[0]
          test_total_acc = test_total_acc + sess_results[1]

        # store
        train_cost_overtime.append(train_total_cost/(len(train_images)/divide_size/batch_size ) ) 
        train_acc_overtime.append(train_total_acc /(len(train_images)/divide_size/batch_size )  )
        test_cost_overtime.append(test_total_cost/(len(test_images)/batch_size ) )
        test_acc_overtime.append(test_total_acc/(len(test_images)/batch_size ) )
        
        # print
        if iter%print_size == 0:
            print('\n=========')
            print("Avg Train Cost: ", train_cost_overtime[-1])
            print("Avg Train Acc: ", train_acc_overtime[-1])
            print("Avg Test Cost: ", test_cost_overtime[-1])
            print("Avg Test Acc: ", test_acc_overtime[-1])
            print('-----------')      

        # shuffle
        if iter%shuffle_size ==  0: 
          print("\n==== shuffling iter: =====",iter," \n")
          train_images,train_labels = shuffle(train_images,train_labels)
          test_images,test_labels = shuffle(test_images,test_labels)
          
        # redeclare
        train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0

  # plot and save
  plt.figure()
  plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r',label="Train")
  plt.plot(range(len(train_cost_overtime)),test_cost_overtime,color='b',label='Test')
  plt.legend()
  plt.title('Cost over time')
  plt.show()

  plt.figure()
  plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='r',label="Train")
  plt.plot(range(len(train_acc_overtime)),test_acc_overtime,color='b',label='Test')
  plt.legend()
  plt.title('Acc over time')
  plt.show()








# ------- end code -----