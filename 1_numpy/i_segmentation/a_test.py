import numpy as np
from skimage import data
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

data = load_digits()
digit = data.data
label = data.target

print(digit.shape)
print(label.shape)


temp = np.reshape(digit[0,:],(8,8))
markers = np.zeros_like(temp)
markers[temp < 1] = 0
markers[temp > 1] = 1

from skimage.filters import sobel

elevation_map = sobel(temp)
from skimage.morphology import watershed
segmentation = watershed(elevation_map, markers)



# plt.imshow(temp,cmap='gray')
# plt.show()


# plt.imshow(segmentation,cmap='gray')
# plt.show()

# plt.imshow(temp*segmentation,cmap='gray')
# plt.show()



mask = (temp>=1) * 1.0
print(mask.shape)
print(temp.shape)

seg = mask * temp

print(mask)
print(segmentation)
print(segmentation-mask)


ss =  (mask > 1) - (temp>1)
print(ss.sum())

# -- end code --