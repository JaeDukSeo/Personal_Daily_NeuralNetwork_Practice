import numpy as np

# Func: Only for 2D convolution 
from scipy.signal import convolve2d

# Func: For Back propagation on Max Pooling
from scipy.ndimage.filters import maximum_filter
import skimage.measure

np.random.seed(12314)


# 0. Declare Sample Matrix and Data
xx = np.random.randn(6,6)
w = np.random.randn(3,3)


# -- end code --