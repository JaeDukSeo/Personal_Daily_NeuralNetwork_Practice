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

training_data = one[:100,:,:]

# 1.75 Training Hyper Parameters
num_epoch = 100
batch_size = 2

if not os.path.exists('images/'):
    os.makedirs('images/')

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# 2. Create the Class
class generator():
    
    def __init__(self):
        
        self.w1 = tf.Variable(xavier_init([13,13,1,3]))
        self.w2 = tf.Variable(xavier_init([11,11,3,5]))
        self.w3 = tf.Variable(xavier_init([9,9,5,7]))

        self.w4 = tf.Variable(xavier_init([7,7,7,7]))
        self.w5 = tf.Variable(xavier_init([5,5,7,5]))

        self.w6 = tf.Variable(xavier_init([3,3,5,3]))
        self.w7 = tf.Variable(xavier_init([1,1,3,1]))
        self.w8 = tf.Variable(xavier_init([1,1,1,1]))

    def getw(self):
        return  [self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8]

    def feed_forward(self,input=None):
        
        l1  = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        l1A = tf.atan(l1) 

        l2 = tf.nn.conv2d(l1A,self.w2,strides=[1,1,1,1],padding='SAME')
        l2A = tf.nn.elu(l2) 

        l3 = tf.nn.conv2d(l2A,self.w3,strides=[1,1,1,1],padding='SAME')
        l3A = tf.atan(l3) 

        l4 = tf.nn.conv2d(l3A,self.w4,strides=[1,1,1,1],padding='SAME')
        l4A = tf.nn.elu(l4)

        l5 = tf.nn.conv2d(l4A,self.w5,strides=[1,1,1,1],padding='SAME')
        l5A = tf.tanh(l5)

        l6 = tf.nn.conv2d(l5A,self.w6,strides=[1,1,1,1],padding='SAME')
        l6A = tf.nn.elu(l6)

        l7 = tf.nn.conv2d(l6A,self.w7,strides=[1,1,1,1],padding='SAME')
        l7A = tf.tanh(l7)

        l8 = tf.nn.conv2d(l7A,self.w8,strides=[1,1,1,1],padding='SAME')

        return l8

class discriminator():
    
    def __init__(self):

        self.w1 = tf.Variable(xavier_init([7,7,1,1]))
        self.w2 = tf.Variable(xavier_init([5,5,1,1]))
        self.w3 = tf.Variable(xavier_init([3,3,1,1]))
        self.w4 = tf.Variable(xavier_init([4096,1024]))
        self.w5 = tf.Variable(xavier_init([1024,1]))
        
    def feed_forward(self,input=None):
        self.input_real = input
        self.layer1_real = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        self.layer1Mean_real = tf.nn.pool(self.layer1_real,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        layer1A = tf_ReLU(self.layer1Mean_real)

        self.layer2_real = tf.nn.conv2d(layer1A,self.w2,strides=[1,1,1,1],padding='SAME')
        self.layer2Mean_real = tf.nn.pool(self.layer2_real ,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        layer2A = tf_ReLU(self.layer2Mean_real)
        
        self.layer3_real = tf.nn.conv2d(layer2A ,self.w3,strides=[1,1,1,1],padding='SAME')
        self.layer3Mean_real = tf.nn.pool(self.layer3_real ,window_shape=[2,2],strides=[2,2],pooling_type="AVG",padding="VALID")
        layer3A = tf_ReLU(self.layer3Mean_real)
        
        self.layer4Input_real = tf.reshape(layer3A ,[batch_size,-1])

        self.layer4_real = tf.matmul(self.layer4Input_real ,self.w4)
        self.layer4A_real = tf_arctan(self.layer4_real )
        
        self.layer5_real = tf.matmul(self.layer4A_real ,self.w5)
        self.layer5A_real = tf_log(self.layer5_real)

        return self.layer5_real

    def getw(self):
        return [self.w1,self.w2,self.w3,self.w4,self.w5]

# 2.5 Make the class 
G = generator()
D = discriminator()

G_w =G.getw()
D_w =D.getw()

# 3. Make the Graph
x_real = tf.placeholder(shape=[None,512,512,1],dtype="float")
x_fake = tf.placeholder(shape=[None,512,512,1],dtype="float")

layer_g = G.feed_forward(x_fake)
layer_D_Fake = D.feed_forward(layer_g)
layer_D_Real = D.feed_forward(x_real)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_D_Real, labels=tf.ones_like(layer_D_Real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_D_Fake, labels=tf.zeros_like(layer_D_Fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_D_Fake, labels=tf.ones_like(layer_D_Fake)))

D_solver = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(D_loss, var_list=D_w)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(G_loss, var_list=G_w)

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

            sess_results1 = sess.run([D_loss,D_solver],feed_dict={x_real:current_image,x_fake:current_data_noise})
            sess_results2 = sess.run([G_loss,G_solver],feed_dict={x_fake:current_data_noise})
            print("Current Iter: ", iter," Current batch : ",current_batch ," current D cost: ",sess_results1[0].sum()," Current G Cost: ",sess_results2[0].sum() ,end='\r')

        # b. Show the Data While Traing
        if iter % 3 == 0:
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