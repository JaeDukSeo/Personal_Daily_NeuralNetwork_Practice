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

    def feed_forward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = tf_tanh(self.layer)
        return self.layerA

    def backpropagation(self,gradient=None):
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


train_images = np.zeros(shape=(128,256,256,1))
train_labels = np.zeros(shape=(128,256,256,1))

for file_index in range(len(train_data)):
    train_images[file_index,:,:]   = np.expand_dims(imresize(imread(train_data[file_index],mode='F',flatten=True),(256,256)),axis=2)
    train_labels[file_index,:,:]   = np.expand_dims(imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(256,256)),axis=2)

train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min())

# --- hyper ---
num_epoch = 100
init_lr = 0.0001
batch_size = 2

# --- make layer ---
# left
l1_1 = conlayer_left(3,1,3)
l1_2 = conlayer_left(3,3,3)
l1_3 = conlayer_left(3,3,3)
    #  * Same as Warshall2 but using BitSets for the matrix rows

l2_1 = conlayer_left(3,3,6)
l2_2 = conlayer_left(3,6,6)
l2_3 = conlayer_left(3,6,6)

l3_1 = conlayer_left(3,6,12)
l3_2 = conlayer_left(3,12,12)
l3_3 = conlayer_left(3,12,12)

l4_1 = conlayer_left(3,12,24)
l4_2 = conlayer_left(3,24,24)
l4_3 = conlayer_left(3,24,24)

l5_1 = conlayer_left(3,24,48)
l5_2 = conlayer_left(3,48,48)
l5_3 = conlayer_left(3,48,24)

# right
l6_1 = conlayer_right(3,24,48)
l6_2 = conlayer_left(3,24,24)
l6_3 = conlayer_left(3,24,12)

l7_1 = conlayer_right(3,12,24)
l7_2 = conlayer_left(3,12,12)
l7_3 = conlayer_left(3,12,6)

l8_1 = conlayer_right(3,6,12)
l8_2 = conlayer_left(3,6,6)
l8_3 = conlayer_left(3,6,3)

l9_1 = conlayer_right(3,3,6)
l9_2 = conlayer_left(3,3,3)
l9_3 = conlayer_left(3,3,3)

l10_final = conlayer_left(3,3,1)

# ---- make graph ----
x = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)

layer1_1 = l1_1.feedforward(x)
layer1_2 = l1_2.feedforward(layer1_1)
layer1_3 = l1_3.feedforward(layer1_2)

layer2_Input = tf.nn.max_pool(layer1_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2_1 = l2_1.feedforward(layer2_Input)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_3 = l2_3.feedforward(layer2_2)

layer3_Input = tf.nn.max_pool(layer2_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3_1 = l3_1.feedforward(layer3_Input)
layer3_2 = l3_2.feedforward(layer3_1)
layer3_3 = l3_3.feedforward(layer3_2)

layer4_Input = tf.nn.max_pool(layer3_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4_1 = l4_1.feedforward(layer4_Input)
layer4_2 = l4_2.feedforward(layer4_1)
layer4_3 = l4_3.feedforward(layer4_2)

layer5_Input = tf.nn.max_pool(layer4_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer5_1 = l5_1.feedforward(layer5_Input)
layer5_2 = l5_2.feedforward(layer5_1)
layer5_3 = l5_3.feedforward(layer5_2)

layer6_Input = tf.concat([layer5_3,layer5_Input],axis=3)
layer6_1 = l6_1.feedforward(layer6_Input)
layer6_2 = l6_2.feedforward(layer6_1)
layer6_3 = l6_3.feedforward(layer6_2)

layer7_Input = tf.concat([layer6_3,layer4_Input],axis=3)
layer7_1 = l7_1.feedforward(layer7_Input)
layer7_2 = l7_2.feedforward(layer7_1)
layer7_3 = l7_3.feedforward(layer7_2)

layer8_Input = tf.concat([layer7_3,layer3_Input],axis=3)
layer8_1 = l8_1.feedforward(layer8_Input)
layer8_2 = l8_2.feedforward(layer8_1)
layer8_3 = l8_3.feedforward(layer8_2)

layer9_Input = tf.concat([layer8_3,layer2_Input],axis=3)
layer9_1 = l9_1.feedforward(layer9_Input)
layer9_2 = l9_2.feedforward(layer9_1)
layer9_3 = l9_3.feedforward(layer9_2)

layer10 = l10_final.feedforward(layer9_3)

cost = tf.reduce_mean(tf.square(layer10-y))
auto_train = tf.train.AdamOptimizer(learning_rate=init_lr).minimize(cost)

# --- start session ---
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        # train
        for current_batch_index in range(0,len(train_images),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([cost,auto_train,layer10],feed_dict={x:current_batch,y:current_label})
            print(' Iter: ', iter, " Cost:  %.32f"% sess_results[0],end='\r')

        print('\n-----------------------')
        train_images,train_labels = shuffle(train_images,train_labels)

        if iter % 2 == 0:
            test_example =   train_images[:2,:,:,:]
            test_example_gt = train_labels[:2,:,:,:]
            sess_results = sess.run([layer10],feed_dict={x:test_example})

            sess_results = sess_results[0][0,:,:,:]
            test_example = test_example[0,:,:,:]
            test_example_gt = test_example_gt[0,:,:,:]

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
                current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
                sess_results = sess.run([cost,auto_train,layer10],feed_dict={x:current_batch,y:current_label})

                plt.figure()
                plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"a_Original Image")
                plt.savefig('gif/'+str(current_batch_index)+"a_Original_Image.png")

                plt.figure()
                plt.imshow(np.squeeze(current_label[0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"b_Original Mask")
                plt.savefig('gif/'+str(current_batch_index)+"b_Original_Mask.png")
                
                plt.figure()
                plt.imshow(np.squeeze(sess_results[2][0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"c_Generated Mask")
                plt.savefig('gif/'+str(current_batch_index)+"c_Generated_Mask.png")

                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"d_Original Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"d_Original_Image_Overlay.png")
            
                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[2][0,:,:,:])),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"e_Generated Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"e_Generated_Image_Overlay.png")

                plt.close('all')


# -- end code --