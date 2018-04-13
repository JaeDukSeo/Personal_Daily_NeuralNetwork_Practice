import numpy as np
import os,sys
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import scipy 
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import cv2

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


# 1. DoG
for x in range(len(one)):
    s = 2
    w = 5
    t = (((w - 1)/2)-0.5)/s
    temp = scipy.ndimage.filters.gaussian_filter(
        one[x,:,:],
        sigma = s,truncate = t
        )

    s = 2
    w = 3
    t = (((w - 1)/2)-0.5)/s
    temp2 = scipy.ndimage.filters.gaussian_filter(
        one[x,:,:],
        sigma = s,truncate = t
        )

    result = temp - temp2
    result = (result > result.mean()-3)*result
    plt.axis('off')
    plt.imshow(result,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()

# 2. DoG 2 
for x in range(len(one)):
    temp = scipy.ndimage.filters.gaussian_filter(
        one[x,:,:],
        sigma = 100
        )
    temp2 = scipy.ndimage.filters.gaussian_filter(
        one[x,:,:],
        sigma = 1
        )

    result = temp - temp2
    result = (result > result.mean())*result
    plt.axis('off')
    plt.imshow(result,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()


# 3. LoG (Smooth)
for x in range(len(one)):
    s = 1.5
    temp = scipy.ndimage.filters.gaussian_filter(
        one[x,:,:],
        sigma = s
        )
    
    lap_kernel = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ])

    result = cv2.filter2D(temp,-1,lap_kernel) 
    plt.axis('off')
    plt.imshow(result,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()

# 4. LoG (no Smooth)
for x in range(len(one)):
    lap_kernel = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ])

    result = cv2.filter2D(one[x,:,:],-1,lap_kernel) 
    plt.axis('off')
    plt.imshow(result,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()




# -- end code --