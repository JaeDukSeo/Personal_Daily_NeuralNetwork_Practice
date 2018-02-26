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

training_data = one
# training_data = np.vstack((one,two,three))

# 1.75 Training Hyper Parameters
num_epoch = 1
batch_size = 20

# 2. Create the Class
class generator():
    
    def __init__(self):
        
        self.w1 = tf.Variable(tf.random_normal([7,7,1,3]))
        self.w2 = tf.Variable(tf.random_normal([5,5,3,5]))
        self.w3 = tf.Variable(tf.random_normal([3,3,5,7]))
        self.w4 = tf.Variable(tf.random_normal([1,1,7,1]))

        self.layer1,self.layer2,self.layer3,self.layer4 = None,None,None,None
        self.layer1A,self.layer2A,self.layer3A,self.layer4A = None,None,None,None
        
        self.input = None

    def feed_forward(self,input=None):
        
        self.input = input
        
        self.layer1 = tf.nn.conv2d(self.input,self.w1,strides=[1,1,1,1],padding="SAME")
        self.layer1A = tf_ReLU(self.layer1 )

        self.layer2 = tf.nn.conv2d(self.layer1A,self.w2,strides=[1,1,1,1],padding="SAME")
        self.layer2A = tf_arctan(self.layer2 )

        self.layer3 = tf.nn.conv2d(self.layer2A,self.w3,strides=[1,1,1,1],padding="SAME")
        self.layer3A = tf_ReLU(self.layer3 )

        self.layer4 = tf.nn.conv2d(self.layer3A,self.w4,strides=[1,1,1,1],padding="SAME")
        self.layer4A =self.output = tf_log(self.layer4 )

        return self.output

    def backprop(self,gradient = None):
        print(8)

class discriminator():
    
    def __init__(self):

        self.w1 = tf.Variable(tf.random_normal([7,7,1,2]))
        self.w2 = tf.Variable(tf.random_normal([5,5,2,3]))
        self.w3 = tf.Variable(tf.random_normal([3,3,3,3]))

        self.w4 = tf.Variable(tf.random_normal([12288,4096]))
        self.w5 = tf.Variable(tf.random_normal([4096,2048]))
        self.w6 = tf.Variable(tf.random_normal([2048,1024]))
        self.w7 = tf.Variable(tf.random_normal([1024,1]))
        
        self.input,self.output = None,None

        self.layer1,self.layer2,self.layer3 = None,None,None
        self.layer1Mean,self.layer2Mean,self.layer3Mean = None,None,None
        self.layer4Input = None

        self.layer4,self.layer5,self.layer6,self.layer7 = None,None,None,None
        self.layer4A,self.layer5A,self.layer6A,self.layer7A = None,None,None,None

    def feed_forward(self,input=None):
        
        self.input = input
        self.layer1 = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        self.layer1Mean = tf.nn.pool(self.layer1,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        
        self.layer2 = tf.nn.conv2d(self.layer1Mean,self.w2,strides=[1,1,1,1],padding='SAME')
        self.layer2Mean = tf.nn.pool(self.layer2,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        
        self.layer3 = tf.nn.conv2d(self.layer2Mean,self.w3,strides=[1,1,1,1],padding='SAME')
        self.layer3Mean = tf.nn.pool(self.layer3,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        
        self.layer4Input = tf.reshape(self.layer3Mean,[batch_size,-1])

        self.layer4 = tf.matmul(self.layer4Input,self.w4)
        self.layer4A = tf_arctan(self.layer4)
        
        self.layer5 = tf.matmul(self.layer4A,self.w5)
        self.layer5A = tf_arctan(self.layer5)
        
        self.layer6 = tf.matmul(self.layer5A,self.w6)
        self.layer6A = tf_arctan(self.layer6)

        self.layer7 = tf.matmul(self.layer6A,self.w7)
        self.layer7A = tf_log(self.layer7)

        return self.layer7A 

    def backprop_real(self,gradient=None):

        return 8

    def backprop_fake(self,gradient=None):
        
        return 8

# 2.5 Make the class 
G = generator()
D = discriminator()

# 3. Make the Graph
x_real = tf.placeholder(shape=[None,512,512,1],dtype="float")
x_fake = tf.placeholder(shape=[None,512,512,1],dtype="float")
y_label= tf.placeholder(shape=[1],dtype="float")

layer_g = G.feed_forward(x_fake)
layer_D_Fake = D.feed_forward(layer_g)
layer_D_Real = D.feed_forward(x_real)

cost = -1.0 * tf_log(layer_D_Real) + tf_log(1.0-layer_D_Fake)

D_BackProp = D.backprop(-1.0 * 1/(layer_D_Real),1.0/(1.0-layer_D_Fake))


# 4. Train Via loop
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        for current_batch in range(0,len(training_data),batch_size):
            
            current_image = training_data[current_batch:current_batch+batch_size,:,:]
            current_data_noise =  current_image + 0.3 * current_image.max() *np.random.randn(current_image.shape[0],current_image.shape[1],current_image.shape[2])

            # current_image      = tf.cast(np.expand_dims(current_image,axis=3),tf.float32) 
            # current_data_noise = tf.cast(np.expand_dims(current_data_noise,axis=3),tf.float32)           

            current_image      = float32(np.expand_dims(current_image,axis=3)) 
            current_data_noise = float32(np.expand_dims(current_data_noise,axis=3))

            sess_results = sess.run(cost,feed_dict={x_real:current_image,x_fake:current_data_noise})


            sys.exit()

# -- end code --