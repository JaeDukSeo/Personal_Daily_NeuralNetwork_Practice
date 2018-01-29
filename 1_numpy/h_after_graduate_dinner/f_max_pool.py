from skimage.measure import block_reduce
import numpy as np
image = np.random.randn(4,4)

image = np.array([
    [1,2,3,4],
    [5,6,7,8],
    [8,10,11,12],
    [8,10,11,14]    
])

print(image)
d = block_reduce(image, block_size=(2,2), func=np.mean)
print(d)

from scipy.ndimage.filters import maximum_filter

arr = image
print(d)

ss = arr*(arr == maximum_filter(arr,footprint=np.ones((2,2))))

print(arr ==maximum_filter(arr,footprint=np.ones((1,2))))
