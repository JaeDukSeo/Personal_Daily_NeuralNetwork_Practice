from PIL import Image
import numpy as np,os,sys,cv2
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

dog_array = np.zeros((24,100,100,3))
store_index = 0
for element in dog_gif:
    img = Image.open(element) 
    for iter in range(img.n_frames):
        img.seek(iter)
        new_frame = np.array(img.convert('RGB')).astype(np.uint8)
        dog_array[store_index,:,:,:] = new_frame
        store_index = store_index + 1

for iter in range(len(dog_array)):
    plt.title(str(iter) + " : dog")
    plt.imshow(np.uint8(dog_array[iter,:,:,:]))
    plt.pause(0.07)

baby_array = np.zeros((24,100,100,3))
store_index = 0
for element in baby_gif:
    img = Image.open(element) 
    for iter in range(img.n_frames):
        img.seek(iter)
        new_frame = np.array(img.convert('RGB'))
        baby_array[store_index,:,:,:] = new_frame
        store_index = store_index + 1

for iter in range(len(baby_array)):
    plt.title(str(iter) + " : baby")
    plt.imshow(np.uint8(baby_array[iter,:,:,:]))
    plt.pause(0.07)
# -- end code --