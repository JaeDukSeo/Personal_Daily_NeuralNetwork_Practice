import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


PathDicom = "../z_CIFAR_data/cifar10batchespy/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        print(filename.lower() )
        if not ".html" in filename.lower() and not  ".meta" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

print(lstFilesDCM)
temp = unpickle(lstFilesDCM[0])

print(len(temp[b'batch_label']))
print(len(temp[b'labels']))
print(len(temp[b'data']))
print(len(temp[b'filenames']))

ss = temp[b'data']
print(type(ss[0]))
print(ss[0])


# --- end code --