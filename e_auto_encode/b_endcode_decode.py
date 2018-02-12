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

training_data = np.vstack((one,two,three))
num_epoch = 1
learn_rate = 0.001

# 2. Build Class for Encoder and Decoder
class Encoder():
    
    def __init__(self):
        self.w1 = np.random.randn(7,7)
        self.w2 = np.random.randn(6,6)
        self.w3 = np.random.randn(5,5)
        self.w4 = np.random.randn(3600,1000)

        self.input,self.output = None,None

        self.l1,self.l1A,self.l1M = None, None, None
        self.l2,self.l2A,self.l2M = None, None, None
        self.l3,self.l3A,self.l3M = None, None, None
        
        self.l4Input  = None
        self.l4,self.l4A = None, None

    def feed_forward(self,input):
        
        self.input = input
        self.l1  = convolve2d(input,self.w1,'valid')
        self.l1M = block_reduce(self.l1,(2,2), np.mean)
        self.l1A = tanh(self.l1M)

        self.l2  = convolve2d(self.l1A,self.w2,'valid')
        self.l2M = block_reduce(self.l2,(2,2), np.mean)
        self.l2A = arctan(self.l2M)

        self.l3  = convolve2d(self.l2A,self.w3,'valid')
        self.l3M = block_reduce(self.l3,(2,2), np.mean)
        self.l3A = tanh(self.l3M)

        self.l4Input = np.reshape(self.l3A,(1,-1))
        self.l4 = self.l4Input.dot(self.w4)
        self.l4A = arctan(self.l4)

        return self.l4A

class Decoder():
    
    def __init__(self):
        print("Hey decode")

    def feed_forward(self,input):
        print(5)

# 3. Define Each Layer object
encoder = Encoder()
decoder = Decoder()

# 4. Traing
for iter in range(num_epoch):

    for image_index in range(len(training_data)):
        current_data = training_data[image_index,:,:]

        encoded_vector = encoder.feed_forward(current_data)
        print(encoded_vector.shape)
        
        decoded_image  = decoder.feed_forward(encoded_vector)
        print(encoded_vector.shape)

        sys.exit()

# -- end code --