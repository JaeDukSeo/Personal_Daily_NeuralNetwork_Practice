import os
import numpy as np
# from nibabel.testing import data_path
# example_filename = os.path.join(data_path, 'example_nifti2.nii.gz')
# # zstat1.nii
# example_filename = os.path.join(data_path, 'example_nifti2.nii.gz')
import matplotlib.pyplot as plt

print(os.path.join(__file__))
import nibabel as nib
img = nib.load('facemask.nii.gz')



print(img.shape)
print(type(img))

data = img.get_data()
print(data.shape)

for iter in range(data.shape[0]):
    print(iter)
    plt.imshow(data[iter,:,:],cmap='gray')
    plt.pause(0.4)


sys.exit()

for iter in range(data.shape[2]):
    print(iter)
    plt.imshow(data[:,:,iter],cmap='gray')
    plt.pause(0.4)


# -- end code --