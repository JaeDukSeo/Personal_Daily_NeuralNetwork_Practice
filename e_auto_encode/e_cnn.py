import numpy as np,sys,os
from scipy.signal import convolve
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from matplotlib.pyplot import plot, draw, show,ion
np.random.randn(6789)

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2
def ReLu(x):
    mask = (x>0) * 1.0
    return mask *x
def d_ReLu(x):
    mask = (x>0) * 1.0
    return mask 
def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))
def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1 + x ** 2)
def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()

# 0. Get the list
PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186756880.24_jpg/"
lstFilesDCM1 = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(PathDicom)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM1.append(os.path.join(dirName,filename))

# lstFilesDCM1_sort = sort_nicely(lstFilesDCM1)

PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186757037.38_jpg/"
lstFilesDCM2 = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM2.append(os.path.join(dirName,filename))


PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186757214.54_jpg/"
lstFilesDCM3 = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM3.append(os.path.join(dirName,filename))

# 1. Read the data into Numpy
one = np.zeros((119,512,512))
two = np.zeros((119,512,512))
three = np.zeros((119,512,512))

# 1.5 Transfer All of the Data into array
print('===== READING DATA ========')
for file_index in range(len(lstFilesDCM1)):
    one[file_index,:,:]   = imread(lstFilesDCM1[file_index],mode='F')
    two[file_index,:,:]   = imread(lstFilesDCM2[file_index],mode='F')
    three[file_index,:,:]   = imread(lstFilesDCM3[file_index],mode='F')
print('===== Done READING DATA ========')

training_data = one
# training_data = np.vstack((one,two,three))

# 2. Declare Classes
class convolutional_net():
    
    def __init__(self):
        self.w1 = np.random.randn(1,3,3) * 0.001
        self.b1 = np.random.randn(1,1,1) * 0.001
        
        self.w2 = np.random.randn(1,3,3) * 0.001
        self.b2 = np.random.randn(1,1,1) * 0.001

        self.w3 = np.random.randn(1,3,3) * 0.001
        self.b3 = np.random.randn(1,1,1) * 0.001
        
        self.input, self.output = None, None
        self.l1,self.l1A,self.l2,self.l2A,self.l3 = None,None,None,None,None

    def feed_forward(self,input=None):
        
        self.input = input
        self.l1  = convolve(self.input ,self.w1,mode='same') +   self.b1
        self.l1A = ReLu(self.l1)

        self.l2 = convolve(self.l1A,self.w2,mode='same') + self.b2
        self.l2A = ReLu(self.l2)

        self.l3 = self.output = convolve(self.l2A,self.w3,mode='same') + self.b3

        return self.output

    def case1_backpropagation(self,gradient=None):

        grad_3_part_1 = gradient

        grad_3_part_3_w = np.pad(self.l2A,((0,0),(1,1),(1,1)),'constant')
        grad_3_part_3_b = np.ones((4,grad_3_part_1.shape[1],grad_3_part_1.shape[2]))
        grad_3_w = np.rot90(convolve(grad_3_part_3_w,np.rot90( grad_3_part_1 ,2),'valid')   ,2)
        grad_3_b = np.rot90(convolve(grad_3_part_3_b,np.rot90( grad_3_part_1 ,2),'valid')   ,2)

        grad_2_part_1 = convolve(self.w3,np.rot90( np.pad(grad_3_part_1,((0,0),(1,1),(1,1)),'constant') ,2),'valid'   )
        grad_2_part_2 = d_ReLu(self.l2)
        grad_2_part_w = np.pad(self.l1A,((0,0),(1,1),(1,1)),'constant')
        grad_2_part_b = np.ones((4,grad_2_part_1.shape[1],grad_2_part_1.shape[2]))
        grad_2_w = np.rot90(convolve(grad_2_part_w,np.rot90( grad_2_part_1 * grad_2_part_2 ,2),'valid')   ,2)
        grad_2_b = np.rot90(convolve(grad_2_part_b,np.rot90( grad_2_part_1 * grad_2_part_2 ,2),'valid')   ,2)

        grad_1_part_1 = convolve(self.w2,np.rot90( np.pad(grad_2_part_1 * grad_2_part_2,((0,0),(1,1),(1,1)),'constant') ,2),'valid'   )
        grad_1_part_2 = d_ReLu(self.l1)
        grad_1_part_w = np.pad(self.input,((0,0),(1,1),(1,1)),'constant')
        grad_1_part_b = np.ones((4,grad_1_part_1.shape[1],grad_1_part_1.shape[2]))
        grad_1_w = np.rot90(convolve(grad_1_part_w,np.rot90( grad_1_part_1 * grad_1_part_2 ,2),'valid')   ,2)
        grad_1_b = np.rot90(convolve(grad_1_part_b,np.rot90( grad_1_part_1 * grad_1_part_2 ,2),'valid')   ,2)

        self.w3 = self.w3 - learning_rate_1 * grad_3_w
        self.b3 = self.b3 - learning_rate_1 * grad_3_b     

        self.w2 = self.w2 - learning_rate_1 * grad_2_w      
        self.b2 = self.b2 - learning_rate_1 * grad_2_b     

        self.w1 = self.w1 - learning_rate_1 * grad_1_w      
        self.b1 = self.b1 - learning_rate_1 * grad_1_b       
                
    def case2_Google_Brain_noise(self,gradient=None,iter=None):
        
        grad_3_part_1 = gradient

        grad_3_part_3_w = np.pad(self.l2A,((0,0),(1,1),(1,1)),'constant')
        grad_3_part_3_b = np.ones((4,grad_3_part_1.shape[1],grad_3_part_1.shape[2]))
        grad_3_w = np.rot90(convolve(grad_3_part_3_w,np.rot90( grad_3_part_1 ,2),'valid')   ,2)
        grad_3_b = np.rot90(convolve(grad_3_part_3_b,np.rot90( grad_3_part_1 ,2),'valid')   ,2)

        grad_2_part_1 = convolve(self.w3,np.rot90( np.pad(grad_3_part_1,((0,0),(1,1),(1,1)),'constant') ,2),'valid'   )
        grad_2_part_2 = d_ReLu(self.l2)
        grad_2_part_w = np.pad(self.l1A,((0,0),(1,1),(1,1)),'constant')
        grad_2_part_b = np.ones((4,grad_2_part_1.shape[1],grad_2_part_1.shape[2]))
        grad_2_w = np.rot90(convolve(grad_2_part_w,np.rot90( grad_2_part_1 * grad_2_part_2 ,2),'valid')   ,2)
        grad_2_b = np.rot90(convolve(grad_2_part_b,np.rot90( grad_2_part_1 * grad_2_part_2 ,2),'valid')   ,2)

        grad_1_part_1 = convolve(self.w2,np.rot90( np.pad(grad_2_part_1 * grad_2_part_2,((0,0),(1,1),(1,1)),'constant') ,2),'valid'   )
        grad_1_part_2 = d_ReLu(self.l1)
        grad_1_part_w = np.pad(self.input,((0,0),(1,1),(1,1)),'constant')
        grad_1_part_b = np.ones((4,grad_1_part_1.shape[1],grad_1_part_1.shape[2]))
        grad_1_w = np.rot90(convolve(grad_1_part_w,np.rot90( grad_1_part_1 * grad_1_part_2 ,2),'valid')   ,2)
        grad_1_b = np.rot90(convolve(grad_1_part_b,np.rot90( grad_1_part_1 * grad_1_part_2 ,2),'valid')   ,2)

        # ------ Calculate The Additive Noise -------
        n_value = 0.01
        ADDITIVE_NOISE_STD = n_value / (np.power((1 + iter), 0.55))
        ADDITIVE_GAUSSIAN_NOISE = np.random.normal(loc=0,scale=ADDITIVE_NOISE_STD)
        # ------ Calculate The Additive Noise -------

        self.w3 = self.w3 - learning_rate_2 * (grad_3_w * ADDITIVE_GAUSSIAN_NOISE)
        self.b3 = self.b3 - learning_rate_2 * (grad_3_b * ADDITIVE_GAUSSIAN_NOISE)     

        self.w2 = self.w2 - learning_rate_2 * (grad_2_w * ADDITIVE_GAUSSIAN_NOISE)      
        self.b2 = self.b2 - learning_rate_2 * (grad_2_b * ADDITIVE_GAUSSIAN_NOISE)     

        self.w1 = self.w1 - learning_rate_2 * (grad_1_w * ADDITIVE_GAUSSIAN_NOISE)      
        self.b1 = self.b1 - learning_rate_2 * (grad_1_b * ADDITIVE_GAUSSIAN_NOISE)  

    def case3_dilated_backpropagation(self,gradient=None,iter=None):
        
        decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter)
        grad_3_part_1 = gradient

        grad_3_part_3_w = np.pad(self.l2A,((0,0),(1,1),(1,1)),'constant')
        grad_3_part_3_b = np.ones((4,grad_3_part_1.shape[1],grad_3_part_1.shape[2]))
        grad_3_w = np.rot90(convolve(grad_3_part_3_w,np.rot90( grad_3_part_1 ,2),'valid')   ,2)
        grad_3_b = np.rot90(convolve(grad_3_part_3_b,np.rot90( grad_3_part_1 ,2),'valid')   ,2)

        grad_2_part_1 = convolve(self.w3,np.rot90( np.pad(grad_3_part_1,((0,0),(1,1),(1,1)),'constant') ,2),'valid'   )
        grad_2_part_2 = d_ReLu(self.l2)
        grad_2_part_w = np.pad(self.l1A,((0,0),(1,1),(1,1)),'constant')
        grad_2_part_b = np.ones((4,grad_2_part_1.shape[1],grad_2_part_1.shape[2]))
        grad_2_w = np.rot90(convolve(grad_2_part_w,np.rot90( grad_2_part_1 * grad_2_part_2 ,2),'valid')   ,2)
        grad_2_b = np.rot90(convolve(grad_2_part_b,np.rot90( grad_2_part_1 * grad_2_part_2 ,2),'valid')   ,2)

        grad_1_part_1 = convolve(self.w2,np.rot90( np.pad(grad_2_part_1 * grad_2_part_2 + decay_propotoin_rate * grad_3_part_1
        ,((0,0),(1,1),(1,1)),'constant') ,2),'valid'   )
        grad_1_part_2 = d_ReLu(self.l1)
        grad_1_part_w = np.pad(self.input,((0,0),(1,1),(1,1)),'constant')
        grad_1_part_b = np.ones((4,grad_1_part_1.shape[1],grad_1_part_1.shape[2]))
        grad_1_w = np.rot90(convolve(grad_1_part_w,np.rot90( grad_1_part_1 * grad_1_part_2 ,2),'valid')   ,2)
        grad_1_b = np.rot90(convolve(grad_1_part_b,np.rot90( grad_1_part_1 * grad_1_part_2 ,2),'valid')   ,2)

        self.w3 = self.w3 - learning_rate_3 * grad_3_w
        self.b3 = self.b3 - learning_rate_3 * grad_3_b     

        self.w2 = self.w2 - learning_rate_3 * grad_2_w      
        self.b2 = self.b2 - learning_rate_3 * grad_2_b     

        self.w1 = self.w1 - learning_rate_3 * grad_1_w      
        self.b1 = self.b1 - learning_rate_3 * grad_1_b  

    def case4_dilated_Google_Brain_noise(self,gradient=None,iter=None):
        decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter)
        grad_3_part_1 = gradient

        grad_3_part_3_w = np.pad(self.l2A,((0,0),(1,1),(1,1)),'constant')
        grad_3_part_3_b = np.ones((4,grad_3_part_1.shape[1],grad_3_part_1.shape[2]))
        grad_3_w = np.rot90(convolve(grad_3_part_3_w,np.rot90( grad_3_part_1 ,2),'valid')   ,2)
        grad_3_b = np.rot90(convolve(grad_3_part_3_b,np.rot90( grad_3_part_1 ,2),'valid')   ,2)

        grad_2_part_1 = convolve(self.w3,np.rot90( np.pad(grad_3_part_1,((0,0),(1,1),(1,1)),'constant') ,2),'valid'   )
        grad_2_part_2 = d_ReLu(self.l2)
        grad_2_part_w = np.pad(self.l1A,((0,0),(1,1),(1,1)),'constant')
        grad_2_part_b = np.ones((4,grad_2_part_1.shape[1],grad_2_part_1.shape[2]))
        grad_2_w = np.rot90(convolve(grad_2_part_w,np.rot90( grad_2_part_1 * grad_2_part_2 ,2),'valid')   ,2)
        grad_2_b = np.rot90(convolve(grad_2_part_b,np.rot90( grad_2_part_1 * grad_2_part_2 ,2),'valid')   ,2)

        grad_1_part_1 = convolve(self.w2,np.rot90( np.pad(grad_2_part_1 * grad_2_part_2 + decay_propotoin_rate * grad_3_part_1
        ,((0,0),(1,1),(1,1)),'constant') ,2),'valid'   )
        grad_1_part_2 = d_ReLu(self.l1)
        grad_1_part_w = np.pad(self.input,((0,0),(1,1),(1,1)),'constant')
        grad_1_part_b = np.ones((4,grad_1_part_1.shape[1],grad_1_part_1.shape[2]))
        grad_1_w = np.rot90(convolve(grad_1_part_w,np.rot90( grad_1_part_1 * grad_1_part_2 ,2),'valid')   ,2)
        grad_1_b = np.rot90(convolve(grad_1_part_b,np.rot90( grad_1_part_1 * grad_1_part_2 ,2),'valid')   ,2)

        # ------ Calculate The Additive Noise -------
        n_value = 0.01
        ADDITIVE_NOISE_STD = n_value / (np.power((1 + iter), 0.55))
        ADDITIVE_GAUSSIAN_NOISE = np.random.normal(loc=0,scale=ADDITIVE_NOISE_STD)
        # ------ Calculate The Additive Noise -------

        self.w3 = self.w3 - learning_rate_4 * (grad_3_w * ADDITIVE_GAUSSIAN_NOISE)
        self.b3 = self.b3 - learning_rate_4 * (grad_3_b * ADDITIVE_GAUSSIAN_NOISE)     

        self.w2 = self.w2 - learning_rate_4 * (grad_2_w * ADDITIVE_GAUSSIAN_NOISE)      
        self.b2 = self.b2 - learning_rate_4 * (grad_2_b * ADDITIVE_GAUSSIAN_NOISE)     

        self.w1 = self.w1 - learning_rate_4 * (grad_1_w * ADDITIVE_GAUSSIAN_NOISE)      
        self.b1 = self.b1 - learning_rate_4 * (grad_1_b * ADDITIVE_GAUSSIAN_NOISE)  

# 3. Declare the model 
case1_convnet = convolutional_net()
case2_convnet = convolutional_net()
case3_convnet = convolutional_net()
case4_convnet = convolutional_net()

# 4. Hyper Parameters
num_epoch = 100
learning_rate_1 = 0.0000000001
learning_rate_2 = 0.000000001
learning_rate_3 = 0.000000001
learning_rate_4 = 0.000000001

proportion_rate = 0.05
decay_rate = 0.07
total_cost1,total_cost2,total_cost3,total_cost4 = 0,0,0,0
cost_array1,cost_array2,cost_array3,cost_array4 = [],[],[],[]

for iter in range(num_epoch):

    # for image_index in range(len(training_data)):
    for image_index in range(10):

        current_data = np.expand_dims(training_data[image_index,:,:],axis=0)
        current_data_noise =  current_data + 0.3 * current_data.max() *np.random.randn(current_data.shape[1],current_data.shape[2])

        current_data_split =       np.vstack((current_data[:,:256,:256],current_data[:,:256,256:],current_data[:,256:,:256],current_data[:,256:,256:]))
        current_data_noise_split = np.vstack((current_data_noise[:,:256,:256],current_data_noise[:,:256,256:],current_data_noise[:,256:,:256],current_data_noise[:,256:,256:]))

        case1_out = case1_convnet.feed_forward(current_data_noise_split)
        case2_out = case2_convnet.feed_forward(current_data_noise_split)
        case3_out = case3_convnet.feed_forward(current_data_noise_split)
        case4_out = case4_convnet.feed_forward(current_data_noise_split)
        
        cost_case1 = np.square(current_data_split - case1_out).sum() * 0.25
        total_cost1 =+ cost_case1

        cost_case2 = np.square(current_data_split - case2_out).sum() * 0.25
        total_cost2 =+ cost_case2
        
        cost_case3 = np.square(current_data_split - case3_out).sum() * 0.25
        total_cost3 =+ cost_case3
        
        cost_case4 = np.square(current_data_split - case4_out).sum() * 0.25
        total_cost4 =+ cost_case4
        print("Real Time Cost image_index : " +str(image_index)+ " Update iter: " + str(iter)+ " Case 1: ",cost_case1, " Case 2: ",cost_case2, " Case 3: ",cost_case3, " Case 4: ",cost_case4,end='\n')

        case1_convnet.case1_backpropagation(            (current_data_split - case1_out) * 0.125 )
        case2_convnet.case2_Google_Brain_noise(         (current_data_split - case2_out) * 0.125,iter  )
        case3_convnet.case3_dilated_backpropagation(    (current_data_split - case3_out) * 0.125,iter  )
        case4_convnet.case4_dilated_Google_Brain_noise( (current_data_split - case4_out) * 0.125,iter  )
        
    cost_array1.append(total_cost1/(len(training_data)*4))
    cost_array2.append(total_cost2/(len(training_data)*4))
    cost_array3.append(total_cost3/(len(training_data)*4))
    cost_array4.append(total_cost4/(len(training_data)*4))

    if iter%10 == 0 :
        print('============ SAMPLE OUTPUT==================')
        for test_index in range(10):
            current_data = np.expand_dims(training_data[test_index,:,:],axis=0)
            current_data_noise =  current_data + 0.3 * current_data.max() *np.random.randn(current_data.shape[1],current_data.shape[2])

            current_data_split =       np.vstack((current_data[:,:256,:256],current_data[:,:256,256:],current_data[:,256:,:256],current_data[:,256:,256:]))
            current_data_noise_split = np.vstack((current_data_noise[:,:256,:256],current_data_noise[:,:256,256:],current_data_noise[:,256:,:256],current_data_noise[:,256:,256:]))

            case1_out = case1_convnet.feed_forward(current_data_noise_split)
            case2_out = case2_convnet.feed_forward(current_data_noise_split)
            case3_out = case3_convnet.feed_forward(current_data_noise_split)
            case4_out = case4_convnet.feed_forward(current_data_noise_split)

            plt.imshow(np.squeeze(current_data),cmap='gray')
            plt.savefig("sucess/" + str(iter) + "_" + str(test_index) +'_.png', bbox_inches='tight')

            plt.imshow(np.squeeze(current_data_noise),cmap='gray')
            plt.savefig("sucess/" + str(iter)+"_" + str(test_index)+'_noise_.png', bbox_inches='tight')

            temp =  np.concatenate((case1_out[0,:,:],case1_out[2,:,:]),axis=0 )
            temp2 = np.concatenate((case1_out[1,:,:],case1_out[3,:,:]),axis=0 )
            temp3 = np.concatenate((temp,temp2),axis=1 )
            plt.imshow(temp3,cmap='gray')
            plt.savefig("case1/" + str(iter)+"_" + str(test_index)+'_.png', bbox_inches='tight')

            temp =  np.concatenate((case2_out[0,:,:],case2_out[2,:,:]),axis=0 )
            temp2 = np.concatenate((case2_out[1,:,:],case2_out[3,:,:]),axis=0 )
            temp3 = np.concatenate((temp,temp2),axis=1 )
            plt.imshow(temp3,cmap='gray')
            plt.savefig("case2/" + str(iter)+"_" + str(test_index)+'_.png', bbox_inches='tight')

            temp =  np.concatenate((case3_out[0,:,:],case3_out[2,:,:]),axis=0 )
            temp2 = np.concatenate((case3_out[1,:,:],case3_out[3,:,:]),axis=0 )
            temp3 = np.concatenate((temp,temp2),axis=1 )
            plt.imshow(temp3,cmap='gray')
            plt.savefig("case3/" + str(iter)+"_" + str(test_index)+'_.png', bbox_inches='tight')
            
            temp =  np.concatenate((case4_out[0,:,:],case4_out[2,:,:]),axis=0 )
            temp2 = np.concatenate((case4_out[1,:,:],case4_out[3,:,:]),axis=0 )
            temp3 = np.concatenate((temp,temp2),axis=1 )
            plt.imshow(temp3,cmap='gray')
            plt.savefig("case4/" + str(iter)+"_" + str(test_index)+'_.png', bbox_inches='tight')
            

plt.plot(range(len(cost_array1)), cost_array1, color='r',label='case 1')
plt.plot(range(len(cost_array2)), cost_array2, color='g',label='case 2')
plt.plot(range(len(cost_array3)), cost_array3, color='b',label='case 3')
plt.plot(range(len(cost_array4)), cost_array4, color='m',label='case 4')
plt.title("Cost over time Graph")
plt.legend()
plt.show()    

# -- end code --