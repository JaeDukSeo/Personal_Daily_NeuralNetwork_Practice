import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt,os,sys
from sklearn.utils import shuffle
from scipy.ndimage import imread

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
        
        self.w1 = tf.Variable(tf.random_normal([893,4],seed=0))

    def feed_forward(self,input=None):
        
        return 8

class discriminator():
    
    def __init__(self):
        print(7)

    def feed_forward(self,input=None):
        
        return 8


# 2.5 Make the class 
G = generator()
D = discriminator()

# 3. Make the Graph
x = tf.placeholder(shape=[None,512,512,1],dtype="float")
y = tf.placeholder(shape=[None,512,512,1],dtype="float")

layer_g = G.feed_forward(x)
layer_D = D.feed_forward(layer_g)


# 4. Train Via loop
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        for current_batch in range(0,len(training_data),batch_size):
            
            current_image = training_data[current_batch:current_batch+batch_size,:,:]
            current_data_noise =  current_image + 0.3 * current_image.max() *np.random.randn(current_image.shape[0],current_image.shape[1],current_image.shape[2])

            plt.imshow(current_image[0,:,:])
            plt.show()

            plt.imshow(current_data_noise[0,:,:])
            plt.show()

            sys.exit()

# -- end code --