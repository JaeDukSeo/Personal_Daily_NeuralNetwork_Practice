import tensorflow as tf
import numpy as np,sys,os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt

np.random.seed(678)
tf.set_random_seed(5678)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)

def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1-tf_tanh(x) ** 2

def tf_softmax(x): return tf.nn.softmax(x)

# --- make class ---
class CNN_Layer():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.005))

    def feedforward(self,input,stride=1,dilate=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding='SAME')
        self.layerA = tf_relu(self.layer)
        return self.layerA
    
    def backprop(self,gradient):
        
        grad_part_1 = gradient 
        grad_part_2 = d_tf_relu(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.nn.conv2d_backprop_filter(
            input = grad_part_3,
            filter_sizes = self.w.shape,
            out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        grad_pass = tf.nn.conv2d_backprop_input(
            input_sizes=[batch_size] + list(self.input.shape[1:]),
            filter = self.w,
            out_backprop = grad_middle,
            strides=[1,1,1,1],padding='SAME'
        )

        update_w = []
        update_w.append(tf.assign(self.w, self.w - learning_rate * grad))
        return grad_pass,update_w

class FNN_layer():
    def __init__(self,input_dim,hidden_dim):
        self.w = tf.Variable(tf.truncated_normal([input_dim,hidden_dim], stddev=0.005))

    def feedforward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = tf_tanh(self.layer)
        return self.layerA

    def backprop(self,gradient=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input 

        grad_x_mid = tf.multiply(grad_part_1,grad_part_2)
        grad = tf.matmul(tf.transpose(grad_part_3),grad_x_mid)
        grad_pass = tf.matmul(tf.multiply(grad_part_1,grad_part_2),tf.transpose(self.w))

        update_w = []
        update_w.append(tf.assign(self.w, self.w - learning_rate * grad))
        return grad_pass,update_w

# --- get data ---
data_location = "./DRIVE/training/images/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower():  # check whether the file's DICOM
            train_data.append(os.path.join(dirName,filename))

data_location = "./DRIVE/training/1st_manual/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower():  # check whether the file's DICOM
            train_data_gt.append(os.path.join(dirName,filename))


train_images = np.zeros(shape=(120,64,64,1))
train_labels = np.zeros(shape=(120,64,64,1))

for file_index in range(len(train_data)-8):
    train_images[file_index,:,:]   = np.expand_dims(imresize(imread(train_data[file_index],mode='F',flatten=True),(64,64)),axis=2)
    train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(64,64)),axis=2)

train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min())




# --- hyper ---
num_epoch = 100
init_lr = 0.01
batch_size = 10

# --- make layer ---
l1 = CNN_Layer(5,1,10)
l2 = CNN_Layer(5,10,20)

FNN_Input = 64 * 64 * 10
l3 = FNN_layer(FNN_Input,1000)
l4 = FNN_layer(1000,64*64)

# ---- make graph ----
x = tf.placeholder(shape=[None,64,64,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,64*64],dtype=tf.float32)

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1)

layer3_Input = tf.reshape(layer2,[batch_size,-1])
layer3 = l3.feedforward(layer3_Input)
layer4 = l4.feedforward(layer3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer4,labels=y))
# auto_train = tf.train.GradientDescentOptimizer(learning_rate=init_lr).minimize(cost)
auto_train = tf.train.AdamOptimizer(learning_rate=init_lr).minimize(cost)







# --- start session ---
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        # train
        for current_batch_index in range(0,len(train_images),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_label = np.reshape(train_labels[current_batch_index:current_batch_index+batch_size,:,:,:],(batch_size,-1))
            sess_results = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_label})
            print(' Iter: ', iter, " Cost:  %.32f"% sess_results[0],end='\r')

        print('\n-----------------------')
        train_images,train_labels = shuffle(train_images,train_labels)

        if iter % 2 == 0:
            test_example =   train_images[:batch_size,:,:,:]
            test_example_gt = np.reshape(train_labels[:batch_size,:,:,:],(batch_size,-1))
            sess_results = sess.run([layer4],feed_dict={x:test_example})

            sess_results = np.reshape(sess_results[0][0],(64,64))
            test_example = test_example[0,:,:,:]
            test_example_gt =np.reshape( test_example_gt[0],(64,64))

            plt.figure()
            plt.imshow(np.squeeze(test_example),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+'Original Image')
            plt.savefig('train_change/epoch_'+str(iter)+"a_Original_Image.png")

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+'Ground Truth Mask')
            plt.savefig('train_change/epoch_'+str(iter)+"b_Original_Mask.png")

            plt.figure()
            plt.imshow(np.squeeze(sess_results),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+'Generated Mask')
            plt.savefig('train_change/epoch_'+str(iter)+"c_Generated_Mask.png")

            plt.figure()
            plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(test_example_gt)),cmap='gray')
            plt.axis('off')
            plt.title('epoch_'+str(iter)+"Ground Truth Overlay")
            plt.savefig('train_change/epoch_'+str(iter)+"d_Original_Image_Overlay.png")

            plt.figure()
            plt.axis('off')
            plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(sess_results)),cmap='gray')
            plt.title('epoch_'+str(iter)+"Generated Overlay")
            plt.savefig('train_change/epoch_'+str(iter)+"e_Generated_Image_Overlay.png")

            plt.close('all')

        # save image if it is last epoch
        if iter == num_epoch - 1:
            train_images,train_labels = shuffle(train_images,train_labels)
            
            for current_batch_index in range(0,len(train_images),batch_size):
                current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
                current_label = np.reshape(train_labels[current_batch_index:current_batch_index+batch_size,:,:,:],(batch_size,-1))
                sess_results = sess.run([cost,auto_train,layer4],feed_dict={x:current_batch,y:current_label})

                plt.figure()
                plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"a_Original Image")
                plt.savefig('gif/'+str(current_batch_index)+"a_Original_Image.png")

                plt.figure()
                plt.imshow(np.reshape(np.squeeze(current_label[0,:,:,:],(64,64) ) ),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"b_Original Mask")
                plt.savefig('gif/'+str(current_batch_index)+"b_Original_Mask.png")
                
                plt.figure()
                plt.imshow(np.squeeze(np.reshape(sess_results[2][0]),(64,64) ),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"c_Generated Mask")
                plt.savefig('gif/'+str(current_batch_index)+"c_Generated_Mask.png")

                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(np.reshape(current_label[0](64,64) ) )),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"d_Original Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"d_Original_Image_Overlay.png")
            
                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(np.reshape(sess_results[2][0](64,64) ))),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"e_Generated Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"e_Generated_Image_Overlay.png")

                plt.close('all')


# -- end code --