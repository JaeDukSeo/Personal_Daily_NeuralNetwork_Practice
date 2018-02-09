from PIL import Image
import numpy as np,os
from matplotlib import pyplot as plt


Pathgif = "./data/dog_preprocessed/"
dog_gif = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Pathgif):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            dog_gif.append(os.path.join(dirName,filename))


Pathgif = "./data/baby_preprocessed/"
baby_gif = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Pathgif):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            baby_gif.append(os.path.join(dirName,filename))

print(baby_gif)
print(dog_gif)




# -- end code --