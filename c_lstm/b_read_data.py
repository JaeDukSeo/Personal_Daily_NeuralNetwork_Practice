import PIL 
from scipy import ndimage
import os,sys
import numpy as np
from matplotlib import pyplot as plt
import cv2,time
from PIL import Image
np.random.seed(6789)


Pathgif = "./data/dog_preprocessed/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Pathgif):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

print(lstFilesDCM[1])
img = Image.open(lstFilesDCM[2])

print(type(img))


for x in lstFilesDCM:
    
    img = Image.open(x) 

    print(x)
    print(img.n_frames)   

Pathgif = "./data/baby_preprocessed/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Pathgif):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))



for x in lstFilesDCM:
    
    img = Image.open(x) 

    print(x)
    print(img.n_frames)   


sys.exit()
for x in dir(img):
    temp = getattr(img,x)
    print(x , "  :  ", temp)

ss = np.array(img)
print(ss.shape)


for i in range(img.n_frames):
    img.seek(i)
    img.show()

    time.sleep(2)

sys.exit()
img.show()
img.seek(1)
img.show()





sys.exit
plt.imshow(img.data)
plt.show()

img.seek()
plt.imshow(img.data)
plt.show()

# -- end code --