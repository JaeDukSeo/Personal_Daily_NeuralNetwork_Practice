from scipy import ndimage
import numpy as np
from scipy import signal
k = np.array([[1,1,1],[1,1,0],[1,0,0]])

temp = np.ones((3,5,5))

print(temp)
print(np.expand_dims(k,axis=0))
print((np.expand_dims(k,axis=0)).shape)
print((np.repeat(np.expand_dims(k,axis=0),3,axis=0).shape))


print(np.repeat(np.expand_dims(k,axis=0),3,axis=0))


filtered = signal.convolve(temp,np.expand_dims(k,axis=0) , mode='same')

print(filtered.shape)

print(filtered)
# ss = ndimage.convolve(temp, np.repeat(np.expand_dims(k,axis=0),3,axis=0), mode='constant')


# print(ss)
