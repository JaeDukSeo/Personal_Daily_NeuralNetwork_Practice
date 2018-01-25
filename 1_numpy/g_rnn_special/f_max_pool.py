import numpy as np
from scipy.ndimage.filters import maximum_filter
import skimage.measure


arr = np.random.randn(4,4)

print(arr)
print('-------')

ss = arr*(arr == maximum_filter(arr,footprint=np.ones((2,2))))

print(maximum_filter(arr,footprint=np.ones((2,2))))
print('-------')

print(arr==maximum_filter(arr,footprint=np.ones((2,2))))
print('-------')
print(arr*(arr==maximum_filter(arr,footprint=np.ones((2,2)))))


print('-------')
print('-------')
print('-------')
print('-------')


print(ss)
a = np.array([
      [  20,  200,   -5,   23],
      [ -13,  134,  119,  100],
      [ 120,   32,   49,   25],
      [-120,   12,    9,   23]
])
sss = skimage.measure.block_reduce(arr, (2,2), np.max)
print('-------')
print(sss)
print('-------')
print(sss==arr)
print('-------')
print(arr.argmax(axis=0))
# --- enc code ---