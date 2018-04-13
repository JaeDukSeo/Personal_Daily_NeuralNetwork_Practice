import numpy as np
import os,sys
import scipy
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import scipy 
from skimage.segmentation import slic
import cv2
from skimage.segmentation import mark_boundaries


def noisy(image):
    try:
        row,col,ch = image.shape
    except:
        row,col = image.shape
        
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p) * 50
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))* 50
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0
    return out


# 0. Read the images
PathDicom = "./sample/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        lstFilesDCM.append(os.path.join(dirName,filename))


# 16. 
for file_index in range(len(lstFilesDCM)):
    temp = noisy(imread(lstFilesDCM[file_index]))
    result = cv2.medianBlur(temp,5)
    plt.imshow(temp)
    plt.show()
    plt.imshow(result)
    plt.show()
print('-----------------------------')


# 18. 
for file_index in range(len(lstFilesDCM)):
    temp = imread(lstFilesDCM[file_index])
    kernel = np.array([
        [0,0,0],
        [0,1,0],
        [0,0,0]
    ]).astype(np.float32)
    result = cv2.filter2D(temp,-1,kernel)
    subtract = temp - result
    final = temp + subtract
    plt.imshow(temp)
    plt.show()
    plt.imshow(subtract)
    plt.show()
    plt.imshow(final)
    plt.show()

for file_index in range(len(lstFilesDCM)):
    temp = imread(lstFilesDCM[file_index])
    result = cv2.medianBlur(temp,3)
    subtract = temp - result
    final = temp + subtract
    
    plt.imshow(temp)
    plt.show()
    plt.imshow(subtract)
    plt.show()
    plt.imshow(final)
    plt.show()

for file_index in range(len(lstFilesDCM)):
    temp = imread(lstFilesDCM[file_index])
    result = cv2.medianBlur(temp,3)
    final = temp + result
    
    plt.imshow(temp)
    plt.show()
    plt.imshow(final)
    plt.show()
print('-----------------------------')






# ---- end code --


# 1. Guassian Blur
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
    result = (result > result.mean())*result
    plt.axis('off')
    plt.imshow(result,cmap='gray')
    plt.show()
