import numpy as np
import matplotlib.pyplot as plt


imgs_utrecht = np.load('./data/utrecht_flair(128)aug.npy')
mask_utrecht = np.load('./data/utrecht_mask(128)aug.npy')

print(imgs_utrecht.shape)
print(mask_utrecht.shape)

temp  = imgs_utrecht[0,:,:,:]
temp2 = mask_utrecht[0,:,:,:]
for x in range(temp.shape[2]):
    plt.imshow(temp[:,:,x],cmap='gray')
    plt.pause(0.04)

    plt.imshow(temp2[:,:,x],cmap='gray')
    plt.pause(0.04)

# -- end code --