import numpy as np

imgs_utrecht = np.load('utrecht_flair(128)aug.npy')
mask_utrecht = np.load('utrecht_mask(128)aug.npy')

print(imgs_utrecht.shape)
print(mask_utrecht.shape)