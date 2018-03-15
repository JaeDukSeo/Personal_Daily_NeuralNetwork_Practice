import tensorflow as tf
import numpy as np,os,dicom,sys
from scipy import misc
import dicom
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr
from sklearn.utils import shuffle
import shutil
np.random.seed(678)
tf.set_random_seed(678)

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1 / (1 - tf.square(x))

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): tf_log(x) * (1- tf_log(X))

def tf_elu(x): return tf.nn.elu(x)

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLu(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def tf_small_blur(x): 
    filter_tf = tf.expand_dims(tf.expand_dims(tf.Variable(tf.constant([
        [1/4,1/4],
        [1/4,1/4]
    ])),axis=2),axis=3)
    return tf.nn.conv2d(x,filter_tf,strides=[1,1,1,1],padding='SAME')

def tf_blur(x): 
    filter_tf = tf.expand_dims(tf.expand_dims(tf.Variable(tf.constant([
        [1/9,1/9,1/9],
        [1/9,1/9,1/9],
        [1/9,1/9,1/9]
    ])),axis=2),axis=3)
    return tf.nn.conv2d(x,filter_tf,strides=[1,1,1,1],padding='SAME')

def tf_wide_blur(x): 
    filter_tf = tf.expand_dims(tf.expand_dims(tf.Variable(tf.constant([
        [1/5,1/5,1/5,1/5,1/5],
        [1/5,1/5,1/5,1/5,1/5],
        [1/5,1/5,1/5,1/5,1/5],
        [1/5,1/5,1/5,1/5,1/5],
        [1/5,1/5,1/5,1/5,1/5]
    ])),axis=2),axis=3)
    return tf.nn.conv2d(x,filter_tf,strides=[1,1,1,1],padding='SAME')

# Make Class
class FCNN():
    
    def __init__(self,kenerl,inc,outc,act,d_act):
        self.w = tf.Variable(tf.random_normal([kenerl,kenerl,inc,outc]))
        self.b = tf.Variable(tf.random_normal([1,512,512,1]))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))

    def getw(self): return [self.w,self.b]

    def feedforward(self,input,stride_row,stride_col,padding,og_input= None,x_input=None):
        self.input  = input

        if og_input==None :
            self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride_row,stride_col,1],padding=padding) + self.b 
        else:
            self.layer  = tf.nn.conv2d(input*og_input,self.w,strides=[1,stride_row,stride_col,1],padding=padding) + self.b 

        self.layer = tf.nn.dropout(self.layer,0.5)

        if  x_input==None:
            self.layerA = self.act(self.layer)
        else:            
            self.layerA = (self.act(self.layer)*tf_small_blur(x_input) ) + tf_small_blur(x_input)
        return self.layerA

    def backprop(self,gradient=None,og_input=None):
        # grad_part_1 = tf.multiply(gradient, og_input)
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.transpose(tf.transpose(tf.multiply(grad_part_1,grad_part_2),[0,2,1,3]),[0,2,1,3])

        grad = tf.nn.conv2d_backprop_filter(input=grad_part_3,filter_sizes=self.w.shape,
        out_backprop=grad_middle,strides=[1,1,1,1],padding='SAME')

        pass_size = list(self.input.shape[1:])
        pass_on_grad = tf.nn.conv2d_backprop_input(input_sizes=[batch_size]+pass_size,filter=self.w,
        out_backprop=grad_middle,strides=[1,1,1,1],padding='SAME')

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return pass_on_grad,grad_update

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


low_dose_train_og = low_dose[:244+82,:,:,:]
high_dose_train_og = high_dose[:244+82,:,:,:]

# low_dose_cv = low_dose[244:244+82,:,:,:]
# high_dose_cv = high_dose[244:244+82,:,:,:]

low_dose_test = low_dose[244+82:,:,:,:]
high_dose_test = high_dose[244+82:,:,:,:]

# Save Location
save_location = "./images_test/"
if os.path.exists(save_location):
    shutil.rmtree(save_location)
os.makedirs(save_location)

# Hyper Param
learning_rate = 0.05 
num_epoch = 301
batch_size = 10 
print_size= 10

# Make Layer Object
l1 = FCNN(6,1,3,tf_ReLU,d_tf_tanh)
l2 = FCNN(3,3,3,tf_ReLU,d_tf_tanh)
l3 = FCNN(2,6,3,tf_ReLU,d_tf_tanh)
l4 = FCNN(1,6,3,tf_ReLU,d_tf_tanh)
l5 = FCNN(1,3,1,tf_log,d_tf_tanh)

l6 = FCNN(2,3,3,tf_ReLU,d_tf_tanh)
l7 = FCNN(1,3,1,tf_log,d_tf_ReLu)

l1w,l2w,l3w,l4w  = l1.getw(),l2.getw(),l3.getw(),l4.getw()
l5w,l6w,l7w = l5.getw(),l6.getw(),l7.getw()

# Make graph
x = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)

layer1 = l1.feedforward(x,1,1,"SAME",x_input=x)
layer2 = l2.feedforward(layer1,1,1,"SAME",og_input=x,x_input=x)

layer3_Input = tf.concat((layer1,layer2),axis=3)
layer3 = l3.feedforward(layer3_Input,1,1,"SAME",og_input=None,x_input=x)

layer4_Input = tf.concat((layer2,layer3),axis=3)
layer4 = l4.feedforward(layer4_Input,1,1,"SAME",og_input=None,x_input=x)
layer5 = l5.feedforward(layer4,1,1,"SAME",og_input=layer1*layer2*layer3*x,x_input=x)
# layer6 = l6.feedforward(layer5,1,1,"SAME",layer1*layer2*layer3*layer4,x)
# layer7 = l7.feedforward(layer6,1,1,"SAME",layer1*layer2*layer3*layer4*layer5,x)

cost = tf.square(tf.subtract(layer5,y))
cost_PSNR = 10*log10(255.0/tf.sqrt(tf.reduce_mean(cost)))
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=[l1w+l2w+l3w+l4w+l5w])

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    total_cost = 0
    cost_over_time = []

    for iter in range(num_epoch):
        
        low_dose_train,high_dose_train = shuffle(low_dose_train_og,high_dose_train_og)
        low_dose_cv =    low_dose_train[:82,:,:,:]
        high_dose_cv =    high_dose_train[:82,:,:,:]
        
        low_dose_train = low_dose_train[82:,:,:,:]
        high_dose_train = high_dose_train[82:,:,:,:]
        
        for current_batch_index in range(0,len(low_dose_train),batch_size):
            current_batch_low  = low_dose_train[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_high = high_dose_train[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_result = sess.run([cost,auto_train],feed_dict={x:current_batch_low,y:current_batch_high})
            print("Current Iter: ", iter, " current batch: ", current_batch_index, " current Cost : ", sess_result[0].sum(),end='\r')
            total_cost= total_cost + np.sum(sess_result[0])

        if iter%print_size==0:
    
            print('\n=================')
            # CV SET Average PSNR
            total_PSNR = 0
            for sample in range(len(low_dose_cv)):
                current_batch_low  = np.expand_dims(low_dose_cv[sample,:,:,:],axis=0).astype(np.float32)
                current_batch_high  = np.expand_dims(high_dose_cv[sample,:,:,:],axis=0).astype(np.float32)
                
                sess_result = sess.run([layer5,cost_PSNR],feed_dict={x:current_batch_low,y:current_batch_high})
                image=  sess_result[0]
                real_PSNR = compare_psnr(np.squeeze(current_batch_high),np.squeeze(image))
                total_PSNR = total_PSNR + real_PSNR
            print("CV Set: Current iter: ", iter, ' Average Peak Noise: ',total_PSNR/len(low_dose_cv))

            # Test SET Average PSNR
            total_PSNR = 0
            for sample in range(len(low_dose_test)):
                current_batch_low  = np.expand_dims(low_dose_test[sample,:,:,:],axis=0).astype(np.float32)
                current_batch_high  = np.expand_dims(high_dose_test[sample,:,:,:],axis=0).astype(np.float32)
                
                sess_result = sess.run([layer5,cost_PSNR],feed_dict={x:current_batch_low,y:current_batch_high})
                image=  sess_result[0]
                real_PSNR = compare_psnr(np.squeeze(current_batch_high),np.squeeze(image))
                total_PSNR = total_PSNR + real_PSNR
            print("Test Set: Current iter: ", iter, ' Average Peak Noise: ',total_PSNR/len(low_dose_test))
            print('=================')

            # Print out the image
            current_batch_low  = np.expand_dims(low_dose_test[-1,:,:,:],axis=0).astype(np.float32)
            current_batch_high  = np.expand_dims(high_dose_test[-1,:,:,:],axis=0).astype(np.float32)  

            sess_result = sess.run([layer5,cost_PSNR],feed_dict={x:current_batch_low,y:current_batch_high})
            image=  sess_result[0]
            real_PSNR = compare_psnr(np.squeeze(current_batch_high),np.squeeze(image))
            print("==Printing : ", sample, " PSNR : ", sess_result[1]," Real : ",real_PSNR)

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

    
        cost_over_time.append(total_cost)
        total_cost=0




# -- end code --