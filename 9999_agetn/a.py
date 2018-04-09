import tensorflow as tf
import numpy as np,sys,os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import image_slicer
import cv2

# --- get data ---
data_location = "./DRIVE/training/images/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower():  # check whether the file's DICOM
            train_data.append(os.path.join(dirName,filename))

data_location = "./DRIVE/training/1st_manual/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower():  # check whether the file's DICOM
            train_data_gt.append(os.path.join(dirName,filename))


train_images = np.zeros(shape=(128,256,256))
train_labels = np.zeros(shape=(128,256,256))

print(train_images.sum())
print(train_labels.sum())

for file_index in range(len(train_data)):
    train_images[file_index,:,:]   = imresize(imread(train_data[file_index],mode='F',flatten=True),(256,256))
    train_labels[file_index,:,:]   = imresize(imread(train_data_gt[file_index],mode='F',flatten=True),(256,256))

train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min())

print(train_images.sum())
print(train_labels.sum())

q1 = np.zeros(64)
one = train_images[0,:,:]
two = medfilt(one,7)

# 8 * 8 matrix make each tile have 32 pixel
M = two.shape[0]//8
N = two.shape[1]//8

tiles = [two[x:x+M,y:y+N] for x in range(0,two.shape[0],M) for y in range(0,two.shape[1],N)]

for i in range(len(tiles)):
    action_threshold = q1[i] + np.random.randn()
    results = tiles[i] < action_threshold
    reward_one = cv2.bitwise_xor(tiles[i],results) / len(tiles)



# -- end code --