import numpy as np
import os,sys
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import scipy 
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

# 0. Read the images
PathDicom = "./images/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        lstFilesDCM.append(os.path.join(dirName,filename))

# 0. Read the data into Numpy
one = np.zeros((7,512,512))

# 0.5 Transfer All of the Data into array
print('===== READING DATA ========')
for file_index in range(len(lstFilesDCM)):
    one[file_index,:,:]   = imresize(imread(lstFilesDCM[file_index],mode='F',flatten=True),(512,512))
print('===== Done READING DATA ========')


# 1. Guassian Blur
for x in range(len(one)):
    temp = scipy.ndimage.filters.gaussian_filter(
        one[x,:,:],
        sigma = 10
        )
    temp2 = scipy.ndimage.filters.gaussian_filter(
        one[x,:,:],
        sigma = 10
        )

    result = temp - temp2
    plt.axis('off')
    plt.imshow(result,cmap='gray')
    plt.show()









# -- end code --