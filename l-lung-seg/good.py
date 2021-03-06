import tensorflow as tf
import numpy as np,sys,os
from sklearn.utils import shuffle
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from scipy.misc import imresize

np.random.seed(678)
tf.set_random_seed(5678)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)

class conlayer():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.005))
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def feedforward(self,input,stride=1,dilate=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],dilations=[1,dilate,dilate,1],padding='SAME')
        self.layerB = tf.nn.batch_normalization(self.layer,scale=True,offset=True,mean=0.0,variance_epsilon=1e-8,variance=1.0)
        self.layer_add =  self.layer + self.layer
        self.layerA = tf_relu(self.layer_add) + self.layer_add / 2 + self.layer / 2 
        return self.layerA

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
init_lr = 0.001
batch_size = 2

# --- make class ---    
l1 = conlayer(3,1,3)
l2 = conlayer(3,3,3)
l3 = conlayer(3,3,3)
l4 = conlayer(3,3,3)
l5 = conlayer(3,3,3)
l6 = conlayer(3,3,3)
l7 = conlayer(3,3,3)
l8 = conlayer(3,3,3)
l9 = conlayer(3,3,3)
l10 = conlayer(3,3,3)

l11 = conlayer(1,31,1)
l12 = conlayer(1,1,1)
l13 = conlayer(1,1,1)

# --- make graph ---
x = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)

layer1B = tf.nn.batch_normalization(x,scale=True,offset=True,mean=0.0,variance=1.0,variance_epsilon=1e-8)
layer1_Input = layer1B + x

layer1 = l1.feedforward(layer1_Input)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2,dilate=2)
layer4 = l4.feedforward(layer3,dilate=3)
layer5 = l5.feedforward(layer4,dilate=5)
layer6 = l6.feedforward(layer5,dilate=8)
layer7 = l7.feedforward(layer6,dilate=13)
layer8 = l8.feedforward(layer7,dilate=21)
layer9 = l9.feedforward(layer8,dilate=34)
layer10 = l10.feedforward(layer9,dilate=55)

layer_concat = tf.concat([x,layer1,layer2,layer3,layer4,layer5,layer6,layer7,layer8,layer9,layer10],axis=3)
layer_drop = tf.nn.dropout(layer_concat,1.0)

layer11 = l11.feedforward(layer_drop)
layer12 = l12.feedforward(layer11)
layer13 = l13.feedforward(layer12)

cost = tf.reduce_mean(tf.square(layer13-y))
auto_train = tf.train.AdamOptimizer(learning_rate=init_lr).minimize(cost)


# --- start session ---
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        # train
        for current_batch_index in range(0,len(train_images),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_label})
            print(' Iter: ', iter, " Cost:  %.32f"% sess_results[0],end='\r')
        print('\n-----------------------')
        train_images,train_labels = shuffle(train_images,train_labels)

        if iter % 2 == 0:
            test_example =   train_images[:2,:,:,:]
            test_example_gt = train_labels[:2,:,:,:]
            sess_results = sess.run([layer13],feed_dict={x:test_example})

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
                sess_results = sess.run([layer13],feed_dict={x:current_batch,y:current_label})

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
                plt.imshow(np.squeeze(sess_results[0][0,:,:,:]),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"c_Generated Mask")
                plt.savefig('gif/'+str(current_batch_index)+"c_Generated_Mask.png")

                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"d_Original Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"d_Original_Image_Overlay.png")
            
                plt.figure()
                plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[0][0,:,:,:])),cmap='gray')
                plt.axis('off')
                plt.title(str(current_batch_index)+"e_Generated Image Overlay")
                plt.savefig('gif/'+str(current_batch_index)+"e_Generated_Image_Overlay.png")

                plt.close('all')


# -- end code --