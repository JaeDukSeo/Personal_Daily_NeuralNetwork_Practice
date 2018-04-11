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

# 0. Read the images
PathDicom = "./sample/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        lstFilesDCM.append(os.path.join(dirName,filename))


# 1. 
# for file_index in range(len(lstFilesDCM)):
#     print(imread(lstFilesDCM[file_index]).shape)
#     print(lstFilesDCM[file_index])
# print('-----------------------------')

# 2. 
original = np.array([1,0])

one = np.array([
    [1,1],
    [1,1]
])
two = np.array([
    [0,1],
    [1,1]
])
three = np.array([
    [0,-1],
    [1,0]
])
four = np.array([
    [0,1],
    [1,0]
])

# print(one.dot(original))
# print(two.dot(original))
# print(three.dot(original))
# print(four.dot(original))
# print('-----------------------------')


# 3. 
# for file_index in range(len(lstFilesDCM)):
#     temp = imread(lstFilesDCM[file_index])
#     kernel = np.ones((20,20),np.float32)/(20*20)
#     result = cv2.filter2D(temp,-1,kernel)
#     plt.imshow(temp)
#     plt.show()
#     plt.imshow(result)
#     plt.show()
# print('-----------------------------')


# 7. 
for file_index in range(len(lstFilesDCM)):
    temp = imread(lstFilesDCM[file_index])
    try:
        temp[:,:,0] = temp[:,:,0] + 2 * temp[:,:,0].std() * np.random.random(temp[:,:,0].shape)
        temp[:,:,1] = temp[:,:,1] + 2 * temp[:,:,1].std() * np.random.random(temp[:,:,1].shape)
        temp[:,:,2] = temp[:,:,2] + 2 * temp[:,:,2].std() * np.random.random(temp[:,:,2].shape)
    except:
        temp= temp + 2 * temp.std() * np.random.random(temp.shape)
    
    kernel = np.ones((5,5),np.float32)/(5*5)
    result = cv2.filter2D(temp,-1,kernel)
    plt.imshow(temp)
    plt.show()
    plt.imshow(result)
    plt.show()
print('-----------------------------')







# ---- end code --