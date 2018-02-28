import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt,os,sys
from sklearn.utils import shuffle
from scipy.ndimage import imread
from numpy import float32
# -2. Set the Random Seed Values
tf.set_random_seed(789)
np.random.seed(568)

# -1 Tf activation functions
def tf_arctan(x):
    return tf.atan(x)
def d_tf_arctan(x):
    return 1.0/(1+tf.square(x))

def tf_ReLU(x):
    return tf.nn.relu(x)
def d_tf_ReLu(x):
    return tf.cast(tf.greater(x, 0),dtype=tf.float32)

def tf_log(x):
    return tf.sigmoid(x)
def d_tf_log(x):
    return tf.sigmoid(x) * (1.0 - tf.sigmoid(x))

# 0. Get the list
PathDicom = "../lung_data_1/"
lstFilesDCM1 = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(PathDicom)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM1.append(os.path.join(dirName,filename))

# 1. Read the data into Numpy
one = np.zeros((119,512,512))

# 1.5 Transfer All of the Data into array
print('===== READING DATA ========')
for file_index in range(len(lstFilesDCM1)):
    one[file_index,:,:]   = imread(lstFilesDCM1[file_index],mode='F')
print('===== Done READING DATA ========')

training_data = one[:110,:,:]

# 1.75 Training Hyper Parameters
num_epoch = 100
batch_size = 2

if not os.path.exists('images/'):
    os.makedirs('images/')

# 2. Create the Class
class generator():
    
    def __init__(self):
        
        self.w1 = tf.Variable(tf.random_normal([7,7,1,3]))
        self.w2 = tf.Variable(tf.random_normal([5,5,3,5]))
        self.w3 = tf.Variable(tf.random_normal([3,3,5,7]))

        self.w4 = tf.Variable(tf.random_normal([28672,1000]))
        self.w5 = tf.Variable(tf.random_normal([1000,28672]))

        self.w6 = tf.Variable(tf.random_normal([3,3,5,7]))
        self.w7 = tf.Variable(tf.random_normal([5,5,3,5]))
        self.w8 = tf.Variable(tf.random_normal([7,7,1,3]))

    def feed_forward(self,input=None):
        
        l1  = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        l1M = tf.nn.pool(l1,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        l1A = tf_arctan(l1M) 

        l2 = tf.nn.conv2d(l1A,self.w2,strides=[1,1,1,1],padding='SAME')
        l2M = tf.nn.pool(l2,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        l2A = tf_arctan(l2M) 

        l3 = tf.nn.conv2d(l2A,self.w3,strides=[1,1,1,1],padding='SAME')
        l3M = tf.nn.pool(l3,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        l3A = tf_arctan(l3M) 

        l4Input = tf.reshape(l3A,[batch_size,-1])
        l4  = tf.matmul(l4Input,self.w4)
        l4A = tf_ReLU(l4)

        l5  = tf.matmul(l4A,self.w5)
        l5A = tf_arctan(l5)
        l5Output = tf.reshape(l5A,[2,64,64,7])

        l6  = tf.nn.conv2d_transpose(l5Output,self.w6,output_shape=[2,128,128,5],strides=[1,2,2,1],padding='SAME')
        l6A = tf_arctan(l6)

        l7  = tf.nn.conv2d_transpose(l6A,self.w7,output_shape=[2,256,256,3],strides=[1,2,2,1],padding='SAME')
        l7A = tf_arctan(l7)

        l8  = tf.nn.conv2d_transpose(l7A,self.w8,output_shape=[2,512,512,1],strides=[1,2,2,1],padding='SAME')
        l8A = tf_log(l8)
        return l8A, [self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8]

class discriminator():
    
    def __init__(self):

        self.w1 = tf.Variable(tf.random_normal([7,7,1,1]))
        self.w2 = tf.Variable(tf.random_normal([5,5,1,1]))
        self.w3 = tf.Variable(tf.random_normal([3,3,1,1]))
        self.w4 = tf.Variable(tf.random_normal([4096,1024]))
        self.w5 = tf.Variable(tf.random_normal([1024,1]))
        
    def feed_forward(self,input=None):
        self.input_real = input
        self.layer1_real = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        self.layer1Mean_real = tf.nn.pool(self.layer1_real,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        
        self.layer2_real = tf.nn.conv2d(self.layer1Mean_real,self.w2,strides=[1,1,1,1],padding='SAME')
        self.layer2Mean_real = tf.nn.pool(self.layer2_real ,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        
        self.layer3_real = tf.nn.conv2d(self.layer2Mean_real ,self.w3,strides=[1,1,1,1],padding='SAME')
        self.layer3Mean_real = tf.nn.pool(self.layer3_real ,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        
        self.layer4Input_real = tf.reshape(self.layer3Mean_real ,[batch_size,-1])

        self.layer4_real = tf.matmul(self.layer4Input_real ,self.w4)
        self.layer4A_real = tf_arctan(self.layer4_real )
        
        self.layer5_real = tf.matmul(self.layer4A_real ,self.w5)
        self.layer5A_real = tf_log(self.layer5_real)

        return self.layer5A_real,[self.w1,self.w2,self.w3,self.w4,self.w5]

# 2.5 Make the class 
G = generator()
D = discriminator()

# 3. Make the Graph
x_real = tf.placeholder(shape=[None,512,512,1],dtype="float")
x_fake = tf.placeholder(shape=[None,512,512,1],dtype="float")

layer_g,layer_g_w = G.feed_forward(x_fake)
layer_D_Fake,layer_D_Fake_w = D.feed_forward(layer_g)
layer_D_Real,layer_D_Real_w = D.feed_forward(x_real)

cost_D = -1.0 * tf_log(layer_D_Real) + tf_log(1.0-layer_D_Fake)
auto_dif = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost_D,var_list=[layer_D_Fake_w,layer_D_Real_w] )

cost_G = -1.0 * tf_log(layer_D_Fake)
auto_dif2 = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost_G,var_list=layer_g_w)

# 4. Train Via loop
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        # a. Start the Training 
        for current_batch in range(0,len(training_data),batch_size):
            
            current_image = training_data[current_batch:current_batch+batch_size,:,:]
            current_data_noise =  current_image + 0.3 * current_image.max() *np.random.randn(current_image.shape[0],current_image.shape[1],current_image.shape[2])

            current_image      = float32(np.expand_dims(current_image,axis=3)) 
            current_data_noise = float32(np.expand_dims(current_data_noise,axis=3))

            sess_results1 = sess.run([cost_D,auto_dif],feed_dict={x_real:current_image,x_fake:current_data_noise})
            sess_results2 = sess.run([cost_G,auto_dif2],feed_dict={x_real:current_image,x_fake:current_data_noise})
            print("Current Iter: ", iter," Current batch : ",current_batch ," current D cost: ",sess_results1[0].sum()," Current G Cost: ",sess_results2[0].sum() ,end='\r')

        # b. Show the Data While Traing
        if iter % 10 == 0:
            current_image = training_data[:2,:,:]
            current_data_noise =  current_image + 0.3 * current_image.max() *np.random.randn(current_image.shape[0],current_image.shape[1],current_image.shape[2])
            current_image      = float32(np.expand_dims(current_image,axis=3)) 
            current_data_noise = float32(np.expand_dims(current_data_noise,axis=3))
            temp = sess.run(layer_g,feed_dict={x_fake:current_data_noise})
            plt.imshow(np.squeeze(current_image[1,:,:,:]),cmap='gray')
            plt.savefig('images/'+str(iter)+'_og.png')
            plt.imshow(np.squeeze(current_data_noise[1,:,:,:]),cmap='gray')
            plt.savefig('images/'+str(iter)+'_noise.png')
            plt.imshow(np.squeeze(temp[1,:,:,:]),cmap='gray')
            plt.savefig('images/'+str(iter)+'_denoise.png')



# -- end code --