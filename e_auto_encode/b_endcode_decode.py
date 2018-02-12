import numpy as np,dicom,sys,os
from scipy.signal import convolve2d
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
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
PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186756880.24/"
lstFilesDCM1 = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM1.append(os.path.join(dirName,filename))

PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186757037.38/"
lstFilesDCM2 = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM2.append(os.path.join(dirName,filename))


PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186757214.54/"
lstFilesDCM3 = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM3.append(os.path.join(dirName,filename))

# 1. Read the data into Numpy
one = np.zeros((119,512,512))
two = np.zeros((119,512,512))
three = np.zeros((119,512,512))

# 1.5 Transfer All of the Data into array
print('===== READING DATA ========')
for file_index in range(len(lstFilesDCM1)):
    one[file_index,:,:]   = np.array(dicom.read_file(lstFilesDCM1[file_index]).pixel_array)
for file_index in range(len(lstFilesDCM2)):
    two[file_index,:,:]   = np.array(dicom.read_file(lstFilesDCM2[file_index]).pixel_array)
for file_index in range(len(lstFilesDCM3)):
    three[file_index,:,:] = np.array(dicom.read_file(lstFilesDCM3[file_index]).pixel_array)
print('===== Done READING DATA ========')


training_data = np.vstack((one,two,three))
num_epoch = 1
learn_rate_d = 0.0001
learn_rate_e = 0.00007

# 2. Build Class for Encoder and Decoder
class Encoder():
    
    def __init__(self):
        self.w1 = np.random.randn(7,7)
        self.w2 = np.random.randn(5,5)
        self.w3 = np.random.randn(3,3)
        self.w4 = np.random.randn(4096,1000)

        self.input,self.output = None,None

        self.l1,self.l1A,self.l1M = None, None, None
        self.l2,self.l2A,self.l2M = None, None, None
        self.l3,self.l3A,self.l3M = None, None, None
        
        self.l4Input  = None
        self.l4,self.l4A = None, None

    def feed_forward(self,input):
        
        self.input = input
        self.l1  = convolve2d(input,self.w1,'same')
        self.l1M = block_reduce(self.l1,(2,2), np.mean)
        self.l1A = tanh(self.l1M)

        self.l2  = convolve2d(self.l1A,self.w2,'same')
        self.l2M = block_reduce(self.l2,(2,2), np.mean)
        self.l2A = arctan(self.l2M)

        self.l3  = convolve2d(self.l2A,self.w3,'same')
        self.l3M = block_reduce(self.l3,(2,2), np.mean)
        self.l3A = tanh(self.l3M)

        self.l4Input = np.reshape(self.l3A,(1,-1))
        self.l4 = self.l4Input.dot(self.w4)
        self.l4A = self.output = arctan(self.l4)

        return self.output

    def back_propagation(self,gradient):

        grad_4_part_1 = gradient
        grad_4_part_2 = d_arctan(self.l4)
        grad_4_part_3 = self.l4Input
        grad_4 = grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)

        grad_3_part_1 = np.reshape((grad_4_part_1 * grad_4_part_2).dot(self.w4.T),(64,64))
        grad_3_part_2 = d_tanh(self.l3M)
        grad_3_part_M = (grad_3_part_1 * grad_3_part_2).repeat(2,axis=0).repeat(2,axis=1)
        grad_3_part_3 = np.pad(self.l2A,1,'constant')
        grad_3 = np.rot90(convolve2d(grad_3_part_3,    np.rot90( grad_3_part_M ,2),'valid')  ,2)

        grad_2_part_1 = convolve2d( self.w3  , np.rot90(np.pad(grad_3_part_M,1,'constant')    ,2)  ,'valid')
        grad_2_part_2 = d_arctan(self.l2M)
        grad_2_part_M = (grad_2_part_1 * grad_2_part_2).repeat(2,axis=0).repeat(2,axis=1)
        grad_2_part_3 = np.pad(self.l1A,2,'constant')
        grad_2 = np.rot90(convolve2d(grad_2_part_3,    np.rot90( grad_2_part_M ,2),'valid')  ,2)
                
        grad_1_part_1 = convolve2d( self.w2  , np.rot90(np.pad(grad_2_part_M,2,'constant')    ,2)  ,'valid')
        grad_1_part_2 = d_tanh(self.l1M)
        grad_1_part_M = (grad_1_part_1 * grad_1_part_2).repeat(2,axis=0).repeat(2,axis=1)
        grad_1_part_3 = np.pad(self.input,3,'constant')
        grad_1 = np.rot90(convolve2d(grad_1_part_3,    np.rot90( grad_1_part_M ,2),'valid')  ,2)

        self.w4 = self.w4 - learn_rate_e *    grad_4
        self.w3 = self.w3 - learn_rate_e *    grad_3     
        self.w2 = self.w2 - learn_rate_e *    grad_2     
        self.w1 = self.w1 - learn_rate_e *    grad_1     
             
class Decoder():
    
    def __init__(self):
        self.w1 = np.random.randn(1000,4096)
        self.w2 = np.random.randn(3,3)
        self.w3 = np.random.randn(5,5)
        self.w4 = np.random.randn(7,7)

        self.input,self.output = None, None

        self.l1,self.l1A = None,None

        self.l2Input = None
        self.l2,self.l2A,self.l2M = None, None, None
        self.l3,self.l3A,self.l3M = None, None, None
        self.l4,self.l4A,self.l4M = None, None, None

    def feed_forward(self,input):
        
        self.input = input
        
        self.l1 = self.input.dot(self.w1)
        self.l1A = arctan(self.l1)

        self.l2Input = np.reshape(self.l1A,(64,64))
        self.l2M   =   self.l2Input.repeat(2,axis=0).repeat(2,axis=1)
        self.l2    =   convolve2d(self.l2M,self.w2,'same')
        self.l2A   =   arctan(self.l2)

        self.l3M   = self.l2A.repeat(2,axis=0).repeat(2,axis=1)
        self.l3    = convolve2d(self.l3M,self.w3,'same')
        self.l3A   = arctan(self.l3)
        
        self.l4M   = self.l3A.repeat(2,axis=0).repeat(2,axis=1)
        self.l4    = convolve2d(self.l4M,self.w4,'same')
        self.l4A = self.output = log(self.l4)

        return self.output

    def back_propagation(self,gradient):

        grad_4_part_1 = gradient
        grad_4_part_2 = d_log(self.l4)
        grad_4_part_3 = np.pad(self.l4M,3,'constant')
        grad_4 = np.rot90(convolve2d(grad_4_part_3,np.rot90(grad_4_part_1 * grad_4_part_2,2),'valid'),2)

        grad_3_part_1 = convolve2d( self.w4,  np.rot90(np.pad(grad_4_part_1 * grad_4_part_2,3,'constant') ,2), 'valid'  )[::2,::2]  
        grad_3_part_2 = d_arctan(self.l3)
        grad_3_part_3 = np.pad(self.l3M,2,'constant')
        grad_3 = np.rot90(convolve2d(grad_3_part_3,np.rot90(grad_3_part_1 * grad_3_part_2,2),'valid'),2)

        grad_2_part_1 = convolve2d( self.w3,  np.rot90(np.pad(grad_3_part_1 * grad_3_part_2,2,'constant') ,2), 'valid'  )[::2,::2]  
        grad_2_part_2 = d_arctan(self.l2)
        grad_2_part_3 = np.pad(self.l2M,1,'constant')
        grad_2 = np.rot90(convolve2d(grad_2_part_3,np.rot90(grad_2_part_1 * grad_2_part_2,2),'valid'),2)

        grad_1_part_1 = np.reshape(convolve2d( self.w2,  np.rot90(np.pad(grad_2_part_1 * grad_2_part_2,1,'constant') ,2), 'valid'  )[::2,::2],(1,-1))
        grad_1_part_2 = d_arctan(self.l1)
        grad_1_part_3 = self.input
        grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

        grad_passon = (grad_1_part_1 * grad_1_part_2).dot(self.w1.T)

        self.w4 = self.w4 - learn_rate_d * grad_4
        self.w3 = self.w3 - learn_rate_d * grad_3
        self.w2 = self.w2 - learn_rate_d * grad_2
        self.w1 = self.w1 - learn_rate_d * grad_1
        
        return grad_passon

# 3. Define Each Layer object
encoder = Encoder()
decoder = Decoder()

# 4. Training both the encoder and decoder
for iter in range(num_epoch):

    for image_index in range(len(training_data)):
        
        current_data = training_data[image_index,:,:]
        current_data_noise =  current_data + 0.3 * current_data.max() *np.random.randn(current_data.shape[0],current_data.shape[1])

        encoded_vector = encoder.feed_forward(current_data_noise)
        decoded_image  = decoder.feed_forward(encoded_vector)

        naive_cost = np.square(decoded_image - current_data).sum() * 0.5
        print()
        
        gradient = decoder.back_propagation(decoded_image - current_data)
        encoder.back_propagation(gradient)

    sys.exit()

# -- end code --