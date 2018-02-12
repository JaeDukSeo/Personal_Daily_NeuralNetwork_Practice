import numpy as np
import dicom,sys
import os
import numpy
from matplotlib import pyplot, cm



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


print(len(lstFilesDCM1))
print(len(lstFilesDCM2))
print(len(lstFilesDCM3))

sys.exit()

print(len(lstFilesDCM))

RefDs = dicom.read_file(lstFilesDCM[0])

print(dir(RefDs))

temp = np.array(RefDs.pixel_array)

print(temp.shape)

pyplot.imshow(temp,cmap='gray')
pyplot.show()


#  -- end code --