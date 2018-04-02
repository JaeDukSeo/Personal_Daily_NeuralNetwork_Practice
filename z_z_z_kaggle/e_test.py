import dicom
import os,sys
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.image as mpimg


PathDicom = "./b_data/test/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    if 'images' in dirName:
        lstFilesDCM.append(dirName)

print(len(lstFilesDCM))
print(lstFilesDCM[0])
print(type(lstFilesDCM))

lowest_weight = 99999999999999999
lowest_height = 99999999999999999

save_num  = 0
for current_file in lstFilesDCM:
    for dirName, subdirList, fileList in os.walk(current_file):
        
        current_list_image = []
        save_test_file_name = ""
        for filename in fileList:
            if ".png" in filename.lower():  # check whether the file's DICOM
                current_list_image.append(dirName+"\\" +filename )
                save_test_file_name = filename

        # current_all_images = np.expand_dims(scipy.misc.imread(current_list_image[0],'F'),axis=0)
        # current_all_images = mpimg.imread(current_list_image[0])
        current_all_images = scipy.misc.imread(current_list_image[0],'F')

        # plt.imshow(current_all_images,cmap='gray')
        # plt.show()

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
        ax.imshow(current_all_images,cmap='gray')
        plt.savefig("./c_preprocessed_data/test/"+str(save_test_file_name),dpi=height)
        plt.close()

        print('----------')
        save_num = save_num + 1

        # sys.exit()
        # input()

print(lowest_weight)
print(lowest_height)


sys.exit()


# ------------------------------------------------


save_num  = 0
for current_file in lstFilesDCM:
    for dirName, subdirList, fileList in os.walk(current_file):
        
        current_list_image = []
        for filename in fileList:
            if ".png" in filename.lower():  # check whether the file's DICOM
                current_list_image.append(dirName+"\\" +filename )

        # current_all_images = np.expand_dims(scipy.misc.imread(current_list_image[0],'F'),axis=0)
        # current_all_images = mpimg.imread(current_list_image[0])
        current_all_images = scipy.misc.imread(current_list_image[0],'F')

        plt.imshow(current_all_images,cmap='gray')
        plt.show()

        sys.exit()
        for x in range(1,len(current_list_image)):
            # curren_image = np.expand_dims(scipy.misc.imread(current_list_image[x],'F'),axis=0)
            # current_all_images = np.vstack((current_all_images,curren_image))
            # curren_image = mpimg.imread(current_list_image[x])
            curren_image = scipy.misc.imread(current_list_image[x],'F')
            current_all_images = current_all_images + curren_image
        
        real_image = dirName[:-5]+"images"
        real_image_path = ""
        for dirName, subdirList, fileList in os.walk(real_image):
            real_image_path = dirName+"\\"+str(fileList[0])

        real_image_readed = scipy.misc.imread(real_image_path,'F')
        # real_image_readed = mpimg.imread(real_image_path)
        
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
        ax.imshow(current_all_images,cmap='gray')
        plt.savefig("./c_preprocessed_data/mask/"+str(save_num) + '_mask.png',dpi=height)
        plt.close()

        fig = plt.figure()
        fig.set_size_inches(width/height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(real_image_readed,cmap='gray')
        plt.savefig("./c_preprocessed_data/train/"+str(save_num) + '_train.png',dpi=height)
        plt.close()

        print(len(current_list_image))
        print(current_all_images.shape)
        print(dirName[:-5])
        print('----------')
        save_num = save_num + 1

        # sys.exit()
        # input()

print(lowest_weight)
print(lowest_height)








# -- end code --
