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
two = np.zeros((119,512,512))
three = np.zeros((119,512,512))

# 1.5 Transfer All of the Data into array
print('===== READING DATA ========')
for file_index in range(len(lstFilesDCM1)):
    one[file_index,:,:]   = imread(lstFilesDCM1[file_index],mode='F')
    # two[file_index,:,:]   = imread(lstFilesDCM2[file_index],mode='F')
    # three[file_index,:,:]   = imread(lstFilesDCM3[file_index],mode='F')
print('===== Done READING DATA ========')

training_data = one[:100,:,:]
# training_data = np.vstack((one,two,three))

# 1.75 Training Hyper Parameters
num_epoch = 100
batch_size = 5

# 2. Create the Class
class generator():
    
    def __init__(self):
        
        self.w1 = tf.Variable(tf.random_normal([13,13,1,3]))
        self.w2 = tf.Variable(tf.random_normal([11,11,3,5]))
        self.w3 = tf.Variable(tf.random_normal([9,9,5,7]))
        self.w4 = tf.Variable(tf.random_normal([7,7,7,9]))
        self.w5 = tf.Variable(tf.random_normal([5,5,9,9]))
        self.w6 = tf.Variable(tf.random_normal([3,3,9,1]))

        self.layer1,self.layer2,self.layer3,self.layer4 = None,None,None,None
        self.layer1A,self.layer2A,self.layer3A,self.layer4A = None,None,None,None

        self.layer5,self.layer6 = None,None
        self.layer5A,self.layer6A = None,None   

        self.input = None

    def feed_forward(self,input=None):
        
        self.input = input
        
        self.layer1 = tf.nn.conv2d(self.input,self.w1,strides=[1,1,1,1],padding="SAME")
        self.layer1A = tf_ReLU(self.layer1 )

        self.layer2 = tf.nn.conv2d(self.layer1A,self.w2,strides=[1,1,1,1],padding="SAME")
        self.layer2A = tf_arctan(self.layer2 )

        self.layer3 = tf.nn.conv2d(self.layer2A,self.w3,strides=[1,1,1,1],padding="SAME")
        self.layer3A = tf_log(self.layer3 )

        self.layer4 = tf.nn.conv2d(self.layer3A,self.w4,strides=[1,1,1,1],padding="SAME")
        self.layer4A  = tf_arctan(self.layer4 )

        self.layer5 = tf.nn.conv2d(self.layer4A,self.w5,strides=[1,1,1,1],padding="SAME")
        self.layer5A  = tf_ReLU(self.layer5 )

        self.layer6 = tf.nn.conv2d(self.layer5A,self.w6,strides=[1,1,1,1],padding="SAME")
        self.layer6A =self.output = tf_log(self.layer6 )

        return self.output

class discriminator():
    
    def __init__(self):

        self.w1 = tf.Variable(tf.random_normal([7,7,1,2]))
        self.w2 = tf.Variable(tf.random_normal([5,5,2,3]))
        self.w3 = tf.Variable(tf.random_normal([3,3,3,3]))

        self.w4 = tf.Variable(tf.random_normal([12288,4096]))
        self.w5 = tf.Variable(tf.random_normal([4096,1]))
        
        self.input_real,self.output_real = None,None
        self.layer1_real,self.layer2_real,self.layer3_real = None,None,None
        self.layer1Mean_real,self.layer2Mean_real,self.layer3Mean_real = None,None,None
        self.layer4Input_real = None
        self.layer4_real,self.layer5_real   = None,None
        self.layer4A_real,self.layer5A_real = None,None

        self.input_fake,self.output_fake = None,None
        self.layer1_fake,self.layer2_fake,self.layer3_fake = None,None,None
        self.layer1Mean_fake,self.layer2Mean_fake,self.layer3Mean_fake = None,None,None
        self.layer4Input_fake = None
        self.layer4_fake,self.layer5_fake   = None,None
        self.layer4A_fake,self.layer5A_fake = None,None


    def feed_forward_real(self,input=None):
        
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

        return self.layer5A_real 

    def feed_forward_fake(self,input = None):
        
        self.input_fake = input
        self.layer1_fake = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        self.layer1Mean_fake = tf.nn.pool(self.layer1_fake,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        
        self.layer2_fake = tf.nn.conv2d(self.layer1Mean_fake,self.w2,strides=[1,1,1,1],padding='SAME')
        self.layer2Mean_fake = tf.nn.pool(self.layer2_fake ,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        
        self.layer3_fake = tf.nn.conv2d(self.layer2Mean_fake ,self.w3,strides=[1,1,1,1],padding='SAME')
        self.layer3Mean_fake = tf.nn.pool(self.layer3_fake ,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        
        self.layer4Input_fake = tf.reshape(self.layer3Mean_fake ,[batch_size,-1])

        self.layer4_fake = tf.matmul(self.layer4Input_fake ,self.w4)
        self.layer4A_fake = tf_arctan(self.layer4_fake )
        
        self.layer5_fake = tf.matmul(self.layer4A_fake ,self.w5)
        self.layer5A_fake = tf_log(self.layer5_fake)

        return self.layer5A_fake 

    def backprop(self,gradient=None):
        
        return 8,8

# 2.5 Make the class 
G = generator()
D = discriminator()

# 3. Make the Graph
x_real = tf.placeholder(shape=[None,512,512,1],dtype="float")
x_fake = tf.placeholder(shape=[None,512,512,1],dtype="float")
y_label= tf.placeholder(shape=[1],dtype="float")

layer_g = G.feed_forward(x_fake)
layer_D_Fake = D.feed_forward_fake(layer_g)
layer_D_Real = D.feed_forward_real(x_real)

cost_D = -1.0 * tf_log(layer_D_Real) + tf_log(1.0-layer_D_Fake)
auto_dif = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost_D)



# 4. Train Via loop
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        for current_batch in range(0,len(training_data),batch_size):
            
            current_image = training_data[current_batch:current_batch+batch_size,:,:]
            current_data_noise =  current_image + 0.3 * current_image.max() *np.random.randn(current_image.shape[0],current_image.shape[1],current_image.shape[2])

            current_image      = float32(np.expand_dims(current_image,axis=3)) 
            current_data_noise = float32(np.expand_dims(current_data_noise,axis=3))

            sess_results = sess.run([cost,auto_dif],feed_dict={x_real:current_image,x_fake:current_data_noise})
            print("Current Iter: ", iter," Current batch : ",current_batch ," current cost: ",sess_results[0].sum(),end='\r')


        if iter % 10 == 0:
            current_image = training_data[:5,:,:]
            current_data_noise =  current_image + 0.3 * current_image.max() *np.random.randn(current_image.shape[0],current_image.shape[1],current_image.shape[2])

            current_image      = float32(np.expand_dims(current_image,axis=3)) 
            current_data_noise = float32(np.expand_dims(current_data_noise,axis=3))

            temp = sess.run(layer_g,feed_dict={x_fake:current_data_noise})

            plt.imshow(np.squeeze(current_image[1,:,:,:]),cmap='gray')
            plt.show()
            plt.imshow(np.squeeze(temp[1,:,:,:]),cmap='gray')
            plt.show()

# -- end code --