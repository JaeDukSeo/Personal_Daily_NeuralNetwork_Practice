import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt,os,sys
from sklearn.utils import shuffle
from scipy.ndimage import imread
from numpy import float32
# -2. Set the Random Seed Values
tf.set_random_seed(789)
np.random.seed(568)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

config = tf.ConfigProto(gpu_options=gpu_options)

config.gpu_options.allow_growth=True


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

path = 'image2/'
if not os.path.exists(path):
    os.makedirs(path)


# Make Class
class ResCNNLayer():
    
    def __init__(self,kernel_size=None,channel_in=None,channel_out=None,act=None,d_act=None):
        
        self.w = tf.Variable(tf.random_normal([kernel_size,kernel_size,channel_in,channel_out]))

        self.input,self.output = None,None
        self.act,self.d_act = act,d_act

        self.layer = None
        self.layerA = None
        self.layerRes = None

    def feed_forward(self,input=None,og_input=None):
        self.input = input
        self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,1,1,1],padding='SAME')
        self.layerA =  self.act(self.layer)
        # self.layerRes = self.output = tf.add(self.layerA,og_input)
        self.layerRes = self.output = tf.multiply(self.layerA,og_input)
        return self.output

    def backprop(self,gradient=None):
        
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input
        grad = tf.transpose(tf.nn.conv2d(grad_part_3,tf.transpose(grad_part_1*grad_part_2),strides=[1,1,1,1],padding='SAME'))
        return grad

    def getw(self):
        return self.w

# Make the Netwokr
l1 = ResCNNLayer(1,1,1,tf_log,d_tf_log)
l2 = ResCNNLayer(3,1,1,tf_log,d_tf_log)
l3 = ResCNNLayer(5,1,3,tf_log,d_tf_log)

l4 = ResCNNLayer(5,3,3,tf_log,d_tf_log)
l5 = ResCNNLayer(3,3,3,tf_log,d_tf_log)
l6 = ResCNNLayer(1,3,1,tf_ReLU,d_tf_ReLu)

l1w,l2w,l3w,l4w,l5w,l6w = l1.getw(),l2.getw(),l3.getw(),l4.getw(),l5.getw(),l6.getw()

# Make the graph
x = tf.placeholder(shape=[None,512,512,1],dtype="float")
y = tf.placeholder(shape=[None,512,512,1],dtype="float")

layer1 = l1.feed_forward(x,x)
layer2 = l2.feed_forward(layer1,x)
layer3 = l3.feed_forward(layer2,x)

layer4 = l4.feed_forward(layer3,x)
layer5 = l5.feed_forward(layer4,x)
layer6 = l6.feed_forward(layer5,x)

loss = tf.reduce_sum(tf.square(tf.subtract(layer6,y) * 0.5))
auto_dif = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss,var_list=[l1w,l2w,l3w,l4w,l5w,l6w])

# grad_6 = l6.backprop(tf.subtract(layer6,y))


# Make Hyper Parameter
num_epoch = 101
batch_size = 10

# Make the Session
with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        for current_batch_index in range(0,len(training_data),batch_size):

            current_batch = training_data[current_batch_index:current_batch_index+batch_size,:,:]
            current_batch_noise =  current_batch + 0.3 * current_batch.max() *np.random.randn(current_batch.shape[0],current_batch.shape[1],current_batch.shape[2])

            current_batch       = float32(np.expand_dims(current_batch,axis=3)) 
            current_batch_noise = float32(np.expand_dims(current_batch_noise,axis=3))

            auto_results = sess.run([loss,auto_dif],feed_dict={x:current_batch_noise,y:current_batch})
            print("Current Iter: ", iter," Current batch : ",current_batch_index ," Current Loss: ",auto_results[0],end='\r')

        # b. Show the Data While Traing
        if iter % 1 == 0:
            current_image = training_data[:2,:,:]
            current_data_noise =  current_image + 0.3 * current_image.max() *np.random.randn(current_image.shape[0],current_image.shape[1],current_image.shape[2])
            current_image      = float32(np.expand_dims(current_image,axis=3)) 
            current_data_noise = float32(np.expand_dims(current_data_noise,axis=3))
            temp = sess.run(layer6,feed_dict={x:current_data_noise})
            plt.imshow(np.squeeze(current_image[1,:,:,:]),cmap='gray')
            plt.savefig(path+str(iter)+'_og.png')
            plt.imshow(np.squeeze(current_data_noise[1,:,:,:]),cmap='gray')
            plt.savefig(path+str(iter)+'_noise.png')
            plt.imshow(np.squeeze(temp[1,:,:,:]),cmap='gray')
            plt.savefig(path+str(iter)+'_denoise.png')


# -- end code --