import numpy as np
import dicom
import os
import numpy
from matplotlib import pyplot, cm



PathDicom = "./DOI/NoduleLayout_1/1.2.840.113704.1.111.1664.1186756141.2/1.2.840.113704.1.111.4116.1186756880.24/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

print(len(lstFilesDCM))

RefDs = dicom.read_file(lstFilesDCM[0])

print(dir(RefDs))

temp = np.array(RefDs.pixel_array)

print(temp.shape)

pyplot.imshow(temp,cmap='gray')
pyplot.show()


#  -- end code --