import os
import numpy as np
from matplotlib import pyplot, cm
import cv2
from PIL import Image
np.random.seed(6789)


Pathgif = "./data/dog_original/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Pathgif):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

print(lstFilesDCM)

img = Image.open(lstFilesDCM[0])
img.seek(img.tell() + 1)  


img.show()

# -- end code --