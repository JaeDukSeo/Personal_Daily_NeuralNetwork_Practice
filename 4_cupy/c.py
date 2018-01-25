# import numpy as np

# a = np.array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24]])

# sub_shape = (3,3)
# view_shape = tuple(np.subtract(a.shape, sub_shape) + 1) + sub_shape
# print(view_shape)
# strides = a.strides + a.strides

# sub_matrices = np.lib.stride_tricks.as_strided(a,view_shape,strides)
# conv_filter = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# m = np.einsum('ij,ijkl->kl',conv_filter,sub_matrices)

# print(m)

import cupy as np
import numpy as np2
from scipy import signal
import time
a = np.random.randn(6000,6000)

w1 = np.random.randn(3,3)



start = time.time()
grad = np.array(signal.convolve2d(np.asnumpy(a), np.asnumpy(w1), boundary='symm', mode='same'))
end = time.time()
print(end - start)
