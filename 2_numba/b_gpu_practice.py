from numba import cuda
from numba import vectorize

# Func: This is a GPU - Version
@vectorize(['int64(int64, int64)'], target='cuda')
def add_ufunc(x, y):
    return x + y






#  -- end code -- 