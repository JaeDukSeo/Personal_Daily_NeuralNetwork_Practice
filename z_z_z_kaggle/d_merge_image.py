import dicom
import os,sys
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from skimage.transform import resize

PathDicom = "./b_data/train/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    if 'masks' in dirName:
        # lstFilesDCM.append(os.path.join(dirName,subdirList))
        lstFilesDCM.append(dirName)

# lstFilesDCM = lstFilesDCM[:2]

print(len(lstFilesDCM))
print(lstFilesDCM[0])
print(type(lstFilesDCM))

lowest_weight = 99999999999999999
lowest_height = 99999999999999999


for current_file in lstFilesDCM:
    for dirName, subdirList, fileList in os.walk(current_file):
        
        current_list_image = []
        for filename in fileList:
            if ".png" in filename.lower():  # check whether the file's DICOM
                current_list_image.append(dirName+"\\" +filename )

        # current_all_images = np.expand_dims(scipy.misc.imread(current_list_image[0],'F'),axis=0)
        current_all_images = scipy.misc.imread(current_list_image[0],'F')
        
        for x in range(1,len(current_list_image)):
            # curren_image = np.expand_dims(scipy.misc.imread(current_list_image[x],'F'),axis=0)
            # current_all_images = np.vstack((current_all_images,curren_image))
            curren_image = scipy.misc.imread(current_list_image[x],'F')
            current_all_images = current_all_images + curren_image
        
        real_image = dirName[:-5]+"images"
        real_image_path = ""
        for dirName, subdirList, fileList in os.walk(real_image):
            real_image_path = dirName+"\\"+str(fileList[0])

        real_image_readed = scipy.misc.imread(real_image_path,'F')
        real_image_readed  =  resize(real_image_readed, (256, 256), mode='constant', preserve_range=True)    
        current_all_images = resize(current_all_images, (256, 256), mode='constant', preserve_range=True)

        sizes = np.shape(current_all_images)
        height = float(sizes[0])
        width = float(sizes[1])

        temp_h = height
        temp_w = width
 
        if lowest_weight>temp_w:
            lowest_weight = temp_w

        if lowest_height>temp_h:
            lowest_height = temp_h
        
        fig = plt.figure()
        fig.set_size_inches(width/height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(current_all_images, cmap='gray')
        plt.savefig(dirName[:-6]+'merged_mask.png',dpi=height)
        plt.close()

        fig = plt.figure()
        fig.set_size_inches(width/height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(real_image_readed, cmap='gray')
        plt.savefig(dirName[:-6]+'merged_train.png',dpi=height)
        plt.close()

        print(len(current_list_image))
        print(current_all_images.shape)
        print(dirName[:-5])
        print('----------')

        # sys.exit()
        # input()

print(lowest_weight)
print(lowest_height)








# -- end code --
