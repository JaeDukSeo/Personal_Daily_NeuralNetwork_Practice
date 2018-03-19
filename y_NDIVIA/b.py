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

temp = unpickle(lstFilesDCM[0])

print(len(temp[b'batch_label']))
print(len(temp[b'labels']))
print(len(temp[b'data']))
print(len(temp[b'filenames']))

print('--------------------')
batch_label = temp[b'batch_label']
labels = temp[b'labels']
data = temp[b'data']
filenames = temp[b'filenames']


for i in range(0):

    print("batch Label : ",batch_label[i])
    print("Label : ",labels[i])    
    print(data[i])    
    print(filenames[i])    

    temp = np.rot90(np.reshape(data[i],(3,32,32)).T,3)
    plt.imshow(temp)
    plt.show()

print('=============')
print(lstFilesDCM)
batch0 = unpickle(lstFilesDCM[0])
batch1 = unpickle(lstFilesDCM[1])
batch2 = unpickle(lstFilesDCM[2])
batch3 = unpickle(lstFilesDCM[3])
batch4 = unpickle(lstFilesDCM[4])
train_batch = np.vstack((batch0[b'data'],batch1[b'data'],batch2[b'data'],batch3[b'data'],batch4[b'data']))
train_label = np.vstack((batch0[b'labels'],batch1[b'labels'],batch2[b'labels'],batch3[b'labels'],batch4[b'labels']))

test_batch = unpickle(lstFilesDCM[5])[b'data']
test_label = np.expand_dims(np.array(unpickle(lstFilesDCM[5])[b'labels']),axis=0)

print(train_batch.shape)
print(train_label.shape)

print(test_batch.shape)
print(test_label.shape)


        


# --- end code --