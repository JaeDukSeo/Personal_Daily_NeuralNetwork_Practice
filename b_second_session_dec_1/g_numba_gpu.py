from __future__ import division
from numba import cuda
import numpy
import math
import numpy as np
from numba import jit
import math
# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation

# Host code   
data = numpy.ones(256)
threadsperblock = 256
blockspergrid = math.ceil(data.shape[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](data)
print(data)

@jit(nopython=True)
def ex1(x, y, out):
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]


in1 = np.arange(10, dtype=np.float64)
in2 = 2 * in1 + 1
out = np.empty_like(in1)

print('in1:', in1)
print('in2:', in2)

ex1(in1, in2, out)

print('out:', out)


# ---- end code ---