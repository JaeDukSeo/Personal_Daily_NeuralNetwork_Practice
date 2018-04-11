import numpy as np
import os,sys
from scipy.ndimage import imread
from scipy.misc import imresize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import scipy 
from skimage.segmentation import slic
import cv2
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


# 1. Identity
transform_matrix = np.array([
    [1,0,0],
    [0,1,0],
]).astype(np.float32)

for x in range(len(one)):
    temp = cv2.warpAffine(one[x,:,:],transform_matrix,(512,512))
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()


# 2. Translation
transform_matrix = np.array([
    [1,0,100],
    [0,1,100],
]).astype(np.float32)

for x in range(len(one)):
    temp = cv2.warpAffine(one[x,:,:],transform_matrix,(512,512))
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()



# 3. Scaling
transform_matrix = np.array([
    [0.75,0,0],
    [0,1.25,0],
]).astype(np.float32)

for x in range(len(one)):
    temp = cv2.warpAffine(one[x,:,:],transform_matrix,(512,512))
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()


# 4. Shearing
transform_matrix = np.array([
    [0.75,0.25,0],
    [0.25,0.75,0],
]).astype(np.float32)

for x in range(len(one)):
    temp = cv2.warpAffine(one[x,:,:],transform_matrix,(512,512))
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()

# 5. Rotation
transform_matrix = np.array([
    [np.cos(np.pi/2),np.sin(np.pi/2), (1-np.cos(np.pi/2))*256-np.sin(np.pi/2)*256 ],
    [-np.sin(np.pi/2),np.cos(np.pi/2),np.sin(np.pi/2)*256+(1-np.cos(np.pi/2))*256  ],
]).astype(np.float32)


for x in range(len(one)):
    temp = cv2.warpAffine(one[x,:,:],transform_matrix,(512,512))
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()


# 6. Homogeneous
transform_matrix = np.array([
    [1,-1/2,300 ],
    [0,0.9,100],
    [0.001,0.001,1.5],
]).astype(np.float32)


for x in range(len(one)):
    temp = cv2.warpPerspective(one[x,:,:],transform_matrix,(512,512))
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x) + '.png',bbox_inches='tight')
    # plt.show()






# -- end code --