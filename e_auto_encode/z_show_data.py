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
        self.w1 = np.random.randn(1,3,3) * 0.01
        self.b1 = np.random.randn(1,1) * 0.01
        
        self.w2 = np.random.randn(1,3,3) * 0.01
        self.b2 = np.random.randn(1,1) * 0.01

        self.w3 = np.random.randn(1,3,3) * 0.01
        self.b3 = np.random.randn(1,1) * 0.01
        
        self.input, self.output = None, None
        self.l1,self.l1A,self.l2,self.l2A,self.l3,self.l3A = None,None,None,None,None,None

    def feed_forward(self,input=None):
        
        self.l1  = convolve(input,self.w1,mode='same') +   self.b1
        self.l1A = ReLu(self.l1)

        self.l2 = convolve(self.l1A,self.w2,mode='same') + self.b2
        self.l2A = ReLu(self.l2)

        self.l3 = convolve(self.l2A,self.w3,mode='same') + self.b3
        self.l3A = self.output = ReLu(self.l3)

        return self.output

    def backpropagation(self,gradient=None):
        print("Gradient")


# 3. Declare the model 
conv_net = convolutional_net()

# 4. Hyper Parameters
num_epoch = 1




for iter in range(num_epoch):

    for image_index in range(len(training_data)):
        
        current_data = training_data[image_index,:,:]
        current_data_noise =  current_data + 0.3 * current_data.max() *np.random.randn(current_data.shape[0],current_data.shape[1])

        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(current_data_noise[:256,:256],cmap='gray')
        axarr[0, 0].get_xaxis().set_visible(False)
        axarr[0, 0].get_yaxis().set_visible(False)
        
        axarr[0, 1].imshow(current_data_noise[:256,256:],cmap='gray')
        axarr[0, 1].set_title('dfsafdsa')
        axarr[0, 1].get_xaxis().set_visible(False)
        axarr[0, 1].get_yaxis().set_visible(False)
        
        axarr[1, 0].imshow(current_data_noise[256:,:256],cmap='gray')
        axarr[1, 0].get_xaxis().set_visible(False)
        axarr[1, 0].get_yaxis().set_visible(False)

        axarr[1, 1].imshow(current_data_noise[256:,256:],cmap='gray')
        axarr[1, 1].get_xaxis().set_visible(False)
        axarr[1, 1].get_yaxis().set_visible(False)
        plt.show()      

        plt.imshow(current_data_noise,cmap='gray')
        plt.show()

        plt.imshow(current_data,cmap='gray')
        plt.show()

        out = conv_net.feed_forward(current_data)
        cost = np.square(current_data - out).sum() * 0.5
        

# -- end code --