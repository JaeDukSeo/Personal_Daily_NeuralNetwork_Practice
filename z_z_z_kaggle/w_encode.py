# Run-Length Encode and Decode

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import dicom
import os,sys
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.image as mpimg


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

print('----- start -----')


PathDicom = "./b_data/train/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    if 'masks' in dirName:
        # lstFilesDCM.append(os.path.join(dirName,subdirList))
        lstFilesDCM.append(dirName)

lowest_weight = 99999999999999999
lowest_height = 99999999999999999

save_num  = 0
for current_file in lstFilesDCM:
    for dirName, subdirList, fileList in os.walk(current_file):
        
        current_list_image = []
        for filename in fileList:
            if ".png" in filename.lower():  # check whether the file's DICOM
                current_list_image.append(dirName+"\\" +filename )

        current_all_images = scipy.misc.imread(current_list_image[0],'F')
        for x in range(1,len(current_list_image)):
            curren_image = scipy.misc.imread(current_list_image[x],'F')
            current_all_images = current_all_images + curren_image
        
        real_image = dirName[:-5]+"images"
        real_image_path = ""
        for dirName, subdirList, fileList in os.walk(real_image):
            real_image_path = dirName+"\\"+str(fileList[0])

        real_image_readed = scipy.misc.imread(real_image_path,'F')

        temp = rle_encode(real_image_readed)
        for x in temp:
            print(x)

        sys.exit()


        
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




print('----- end -----')


# -- end code -- 