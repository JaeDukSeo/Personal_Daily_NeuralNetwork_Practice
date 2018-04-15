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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
def RLenc(img,order='F',format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    
    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
    
        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs

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
        
        print(RLenc(current_all_images))

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