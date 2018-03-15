import tensorflow as tf
import numpy as np,os,dicom,sys
from scipy import misc
import dicom
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr
from sklearn.utils import shuffle
import shutil
np.random.seed(2)
tf.set_random_seed(2)

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1 / (1 - tf.square(x))

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): tf_log(x) * (1- tf_log(X))

def tf_elu(x): return tf.nn.elu(x)

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLu(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)

# 0. Get the list
PathDicom = "../z_super/sliceTh_0.75_exposure_25/"
lstFilesDCM_low = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM_low.append(os.path.join(dirName,filename))

PathDicom = "../z_super/sliceTh_0.75_exposure_200/"
lstFilesDCM_high = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM_high.append(os.path.join(dirName,filename))

low_dose  = np.zeros((407,512,512,1))
high_dose = np.zeros((407,512,512,1))

# 1.5 Transfer All of the Data into array
print('===== READING / NORMALIZING DATA ========')
for file_index in range(len(lstFilesDCM_low)):
    
    temp = np.expand_dims(dicom.read_file(lstFilesDCM_low[file_index]).pixel_array.astype(np.float32),axis=3)
    temp = (temp - temp.min())/(temp.max() - temp.min())
    low_dose[file_index,:,:,:] = temp

    temp = np.expand_dims(dicom.read_file(lstFilesDCM_high[file_index]).pixel_array.astype(np.float32),axis=3)
    temp = (temp - temp.min())/(temp.max() - temp.min())
    high_dose[file_index,:,:,:] = temp
print('===== READING / NORMALIZING DATA ========')

# Save Location
save_location = "./images_compare/"
if os.path.exists(save_location):
    shutil.rmtree(save_location)
os.makedirs(save_location)

# Make weights
w1 = tf.get_variable('W1', shape=[3,3,1,3], initializer=tf.contrib.layers.xavier_initializer()) 
w2 = tf.get_variable('W2', shape=[3,3,3,3], initializer=tf.contrib.layers.xavier_initializer()) 
w3 = tf.get_variable('W3', shape=[3,3,3,3], initializer=tf.contrib.layers.xavier_initializer()) 
w4 = tf.get_variable('W4', shape=[3,3,3,3], initializer=tf.contrib.layers.xavier_initializer()) 
w5 = tf.get_variable('W5', shape=[3,3,3,3], initializer=tf.contrib.layers.xavier_initializer()) 
w6 = tf.get_variable('W6', shape=[3,3,6,3], initializer=tf.contrib.layers.xavier_initializer()) 
w7 = tf.get_variable('W7', shape=[3,3,6,1], initializer=tf.contrib.layers.xavier_initializer()) 

# Make a graph
x = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)

layer1 = tf.nn.atrous_conv2d(x,w1,padding="SAME",rate=1)
layer1A = tf.nn.relu(layer1)

layer2 = tf.nn.atrous_conv2d(layer1A,w2,padding="SAME",rate=2)
layer2BN = tf.contrib.layers.batch_norm(layer2)
layer2A = tf.nn.relu(layer2BN)

layer3 = tf.nn.atrous_conv2d(layer2A,w3,padding="SAME",rate=3)
layer3BN = tf.contrib.layers.batch_norm(layer3)
layer3A = tf.nn.relu(layer3BN)

layer4 = tf.nn.atrous_conv2d(layer3A,w4,padding="SAME",rate=4)
layer4BN = tf.contrib.layers.batch_norm(layer4)
layer4A = tf.nn.relu(layer4BN)

layer5 = tf.nn.atrous_conv2d(layer4A,w5,padding="SAME",rate=4)
layer5BN = tf.contrib.layers.batch_norm(layer5)
layer5A = tf.nn.relu(layer5BN)

layer6_Input = tf.concat((layer5A,layer2A),axis=3)
layer6 = tf.nn.atrous_conv2d(layer6_Input,w6,padding="SAME",rate=4)
layer6BN = tf.contrib.layers.batch_norm(layer6)
layer6A = tf.nn.relu(layer6BN)

layer7_Input = tf.concat((layer6A,layer1A),axis=3)
layer7 = tf.nn.atrous_conv2d(layer7_Input,w7,padding="SAME",rate=4)
layer7BN = tf.contrib.layers.batch_norm(layer7)
layer7A = tf.nn.relu(layer7BN)

final_layer = tf.add(x,layer7A)

cost = tf.reduce_sum(tf.square(final_layer-y))

auto_train  = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost,var_list=[w1,w2,w3,w4,w5,w6,w7])
auto_train_20  = tf.train.MomentumOptimizer(learning_rate=0.00001,momentum=0.9).minimize(cost,var_list=[w1,w2,w3,w4,w5,w6,w7])

# hyper
num_epoch = 101
batch_size = 10 
print_size= 10

# Run the Session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    total_cost = 0
    cost_over_time = []

    low_does_last = low_dose[-1,:,:,:]
    high_does_last = high_dose[-1,:,:,:]
    
    low_dose = low_dose[:-1,:,:,:]
    high_dose = high_dose[:-1,:,:,:]

    low_dose,high_dose = shuffle(low_dose,high_dose)

    # Get the 80 Percent of the data
    low_dose_train_og =low_dose[:327,:,:,:]
    high_dose_train_og =high_dose[:327,:,:,:]

    # Split the test set 
    low_dose_test=   low_dose[327:,:,:,:]
    high_dose_test = high_dose[327:,:,:,:]

    for iter in range(num_epoch):
        
        low_dose_train,high_dose_train = shuffle(low_dose_train_og,high_dose_train_og)
        low_dose_cv =    low_dose_train[:82,:,:,:]
        high_dose_cv =    high_dose_train[:82,:,:,:]
        
        low_dose_train = low_dose_train[82:,:,:,:]
        high_dose_train = high_dose_train[82:,:,:,:]
        
        for current_batch_index in range(0,len(low_dose_train),batch_size):
            current_batch_low  = low_dose_train[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_high = high_dose_train[current_batch_index:current_batch_index+batch_size,:,:,:]

            if iter <20:    
                sess_result = sess.run([cost,auto_train],feed_dict={x:current_batch_low,y:current_batch_high})
            else:
                sess_result = sess.run([cost,auto_train_20],feed_dict={x:current_batch_low,y:current_batch_high})
                
            print("Current Iter: ", iter, " current batch: ", current_batch_index, " current Cost : ", sess_result[0].sum(),end='\r')
            total_cost= total_cost + np.sum(sess_result[0])

        if iter%print_size==0:
    
            print('\n=================')
            # CV SET Average PSNR
            total_PSNR = 0
            for sample in range(len(low_dose_cv)):
                current_batch_low  = np.expand_dims(low_dose_cv[sample,:,:,:],axis=0).astype(np.float32)
                current_batch_high  = np.expand_dims(high_dose_cv[sample,:,:,:],axis=0).astype(np.float32)
                
                sess_result = sess.run([final_layer],feed_dict={x:current_batch_low,y:current_batch_high})
                image=  sess_result[0]
                real_PSNR = compare_psnr(np.squeeze(current_batch_high),np.squeeze(image))
                total_PSNR = total_PSNR + real_PSNR
            print("CV Set: Current iter: ", iter, ' Average Peak Noise: ',total_PSNR/len(low_dose_cv))

            # Test SET Average PSNR
            total_PSNR = 0
            for sample in range(len(low_dose_test)):
                current_batch_low  = np.expand_dims(low_dose_test[sample,:,:,:],axis=0).astype(np.float32)
                current_batch_high  = np.expand_dims(high_dose_test[sample,:,:,:],axis=0).astype(np.float32)
                
                sess_result = sess.run([final_layer],feed_dict={x:current_batch_low,y:current_batch_high})
                image=  sess_result[0]
                real_PSNR = compare_psnr(np.squeeze(current_batch_high),np.squeeze(image))
                total_PSNR = total_PSNR + real_PSNR
            print("Test Set: Current iter: ", iter, ' Average Peak Noise: ',total_PSNR/len(low_dose_test))

            # Print out the image
            current_batch_low  = np.expand_dims(low_does_last,axis=0).astype(np.float32)
            current_batch_high  = np.expand_dims(high_does_last,axis=0).astype(np.float32)  

            sess_result = sess.run([final_layer],feed_dict={x:current_batch_low,y:current_batch_high})
            image=  sess_result[0]
            real_PSNR = compare_psnr(np.squeeze(current_batch_high),np.squeeze(image))
            print("Final Image to Compare: ", sample," Real : ",real_PSNR)

            plt.figure()
            plt.imshow(np.squeeze(image),cmap='gray')
            plt.savefig(save_location +str(iter) +"_"+str(sample) + "_denoised_"+ str(real_PSNR)+ ".png")

            plt.figure()
            plt.imshow(np.squeeze(current_batch_low),cmap='gray')
            plt.savefig(save_location +str(iter) +"_"+str(sample) + "_noised_"+ str(real_PSNR)+ ".png")

            plt.figure()
            plt.imshow(np.squeeze(current_batch_high),cmap='gray')
            plt.savefig(save_location + str(iter) +"_"+str(sample) + "_groundt_"+ str(real_PSNR)+ ".png")
            plt.close('all')
            print('=================')


    
        cost_over_time.append(total_cost)
        total_cost=0




# -- end code --