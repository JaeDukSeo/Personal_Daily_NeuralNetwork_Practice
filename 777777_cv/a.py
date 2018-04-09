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


# 1. Identity
kernerl = np.array([
    [0,0,0],
    [0,1,0],
    [0,0,0]
])

for x in range(len(one)):
    temp = convolve2d(one[x,:,:],kernerl,mode='same')
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')

# 2. Edge Detection (Horizontal)
kernerl = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

for x in range(len(one)):
    temp = convolve2d(one[x,:,:],kernerl,mode='same')
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')

# 3. Edge Detection (Vertical)
kernerl = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])

for x in range(len(one)):
    temp = convolve2d(one[x,:,:],kernerl,mode='same')
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')

# 4. Gradient Magnitude
kernerl1 = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])
kernerl2 = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

for x in range(len(one)):
    temp1 = convolve2d(one[x,:,:],kernerl1,mode='same')
    temp2 = convolve2d(one[x,:,:],kernerl2,mode='same')
    
    temp3 = np.sqrt(temp1**2 + temp2**2)

    plt.axis('off')
    plt.imshow(temp3,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')


# 5. Gradient Direction
kernerl1 = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])
kernerl2 = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])

for x in range(len(one)):
    temp1 = convolve2d(one[x,:,:],kernerl1,mode='same')
    temp2 = convolve2d(one[x,:,:],kernerl2,mode='same')
    
    temp3 = np.arctan(temp1/temp2)

    plt.axis('off')
    plt.imshow(temp3,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')


# 6. Sobel Gradient Magnitude
kernerl1 = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
])
kernerl2 = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])

for x in range(len(one)):
    temp1 = convolve2d(one[x,:,:],kernerl1,mode='same')
    temp2 = convolve2d(one[x,:,:],kernerl2,mode='same')
    
    temp3 = np.sqrt(temp1**2 + temp2**2)

    plt.axis('off')
    plt.imshow(temp3,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')


# 7. Sobel Gradient Direction
kernerl1 = np.array([
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
])
kernerl2 = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])

for x in range(len(one)):
    temp1 = convolve2d(one[x,:,:],kernerl1,mode='same')
    temp2 = convolve2d(one[x,:,:],kernerl2,mode='same')
    
    temp3 = np.arctan(temp1/temp2)

    plt.axis('off')
    plt.imshow(temp3,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')


# 8. Guassian Blur
for x in range(len(one)):
    temp = scipy.ndimage.filters.gaussian_filter(
        one[x,:,:],
        sigma = 10
        )
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')




# 9. Sharpening
kernerl1 = np.array([
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
])
kernerl2 = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

for x in range(len(one)):
    temp1 = convolve2d(one[x,:,:],kernerl1,mode='same')
    temp2 = convolve2d(one[x,:,:],kernerl2,mode='same')
    
    temp3 = np.sqrt(temp1**2 + temp2**2)

    plt.axis('off')
    plt.imshow(temp3 +one[x,:,:] ,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')



# 10. Emboss
kernerl = np.array([
    [-1,-1,0],
    [-1,0,1],
    [0,1,1]
])

for x in range(len(one)):
    temp = convolve2d(one[x,:,:],kernerl,mode='same') + 128
    plt.axis('off')
    plt.imshow(temp,cmap='gray')
    plt.savefig(str(x)+'.png',bbox_inches='tight')



# 11. Super Pixel
for x in range(len(one)):
    segments = slic(one[x,:,:], n_segments = 50, sigma = 10)
    plt.axis('off')
    plt.imshow(mark_boundaries(one[x,:,:], segments))
    plt.savefig(str(x)+'.png',bbox_inches='tight')




print('done')

# -- end code --