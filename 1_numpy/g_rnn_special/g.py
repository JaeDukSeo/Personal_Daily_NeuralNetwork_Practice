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





temp = np.array([
    [2,3,2,4],
    [8,3,12,1],
    [12,3,4,9],
    [90,2,-8,12]
])


sss = skimage.measure.block_reduce(temp, (2,2), np.max)
print('----------------')
print(temp)
print('----------------')
print(sss)
print('----------------')
print( np.where(temp == temp.max(axis=1) ))
print('----------------')
print('----------------')

#forward
activationPrevious = np.copy(temp)
activations = skimage.measure.block_reduce(temp, block_size=(2,2), func=np.max)

print(activations)

maxs = activations.repeat(2, axis=0).repeat(2, axis=1)
mask = np.equal(activationPrevious, maxs).astype(int)
delta = np.multiply(maxs, mask)

print(delta)




#backward
# delta = delta.repeat(2, axis=2).repeat(2, axis=3)
# delta = np.multiply(delta, mask)


# --- enc code ---