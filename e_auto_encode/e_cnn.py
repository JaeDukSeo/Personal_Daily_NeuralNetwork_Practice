import numpy as np,dicom,sys,os
from scipy.signal import convolve2d
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import re
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



def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

# 0. Get the list
PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186756880.24/"
lstFilesDCM1 = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(PathDicom)):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM1.append(os.path.join(dirName,filename))

lstFilesDCM1_sort = sort_nicely(lstFilesDCM1)

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






for file_index in range(len(lstFilesDCM1_sort)):
    print(lstFilesDCM1_sort[file_index])
    temp = dicom.read_file(lstFilesDCM1_sort[file_index])

    print(dir(temp))
    sys.exit()











sys.exit()
# 1. Read the data into Numpy
one = np.zeros((119,512,512))
# two = np.zeros((119,512,512))
# three = np.zeros((119,512,512))
# 1.5 Transfer All of the Data into array
print('===== READING DATA ========')
for file_index in range(len(lstFilesDCM1_sort)):
    print(lstFilesDCM1_sort[file_index])
    one[file_index,:,:]   = np.array(dicom.read_file(lstFilesDCM1_sort[file_index]).pixel_array)
# for file_index in range(len(lstFilesDCM2)):
#     two[file_index,:,:]   = np.array(dicom.read_file(lstFilesDCM2[file_index]).pixel_array)
# for file_index in range(len(lstFilesDCM3)):
#     three[file_index,:,:] = np.array(dicom.read_file(lstFilesDCM3[file_index]).pixel_array)
print('===== Done READING DATA ========')











sys.exit()
for x in one:
    
    plt.imshow(x,cmap='gray')
    plt.pause(0.3)



# -- end code --