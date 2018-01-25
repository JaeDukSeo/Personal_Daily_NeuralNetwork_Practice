import numpy as np
from scipy.ndimage.filters import maximum_filter
import skimage.measure

np.random.seed(45678)

arr = np.random.randn(4,4)

print(arr)
print(np.where(arr == arr.max()))
print('----------------')

sss = skimage.measure.block_reduce(arr, (2,2), np.max)
print(sss)
print('----------------')

print(maximum_filter(arr,size=4))













# --- enc code ---