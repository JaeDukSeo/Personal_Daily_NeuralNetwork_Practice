import numpy as np
import dicom,sys
import os
import numpy
from matplotlib import pyplot as plt, cm

np.random.randn(6789)

# 0. Get the list
PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186756880.24/"
lstFilesDCM1 = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM1.append(os.path.join(dirName,filename))

PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186757037.38/"
lstFilesDCM2 = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM2.append(os.path.join(dirName,filename))


PathDicom = "./lung_data/DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186757214.54/"
lstFilesDCM3 = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM3.append(os.path.join(dirName,filename))

# 1. Read the data into Numpy
one = np.zeros((119,512,512))


for file_index in range(len(lstFilesDCM1)):
    RefDs = dicom.read_file(lstFilesDCM1[file_index])
    one[file_index,:,:] = np.array(RefDs.pixel_array)
plt.figure(1)

for temp in range(len(one)):
    plt.subplot(121)
    plt.imshow(one[temp,:,:] ,cmap='gray')
    plt.subplot(122)
    plt.imshow(one[temp,:,:] + 0.3 * one[temp,:,:].max() *np.random.randn(512,512),cmap='gray')
    plt.pause(0.9)


sys.exit()

print(len(lstFilesDCM))

RefDs = dicom.read_file(lstFilesDCM[0])

print(dir(RefDs))

temp = np.array(RefDs.pixel_array)

print(temp.shape)

pyplot.imshow(temp,cmap='gray')
pyplot.show()


#  -- end code --