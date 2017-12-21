import numpy as np
from sklearn.datasets import load_sample_images
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from scipy import signal

np.random.seed(1)

# 0. Load the Sample data set - Choose between digit and sample
# dataset = load_sample_images() 
dataset = load_digits() 
current_img =dataset.images[0]

# 1. Display the original data
plt.imshow(current_img,cmap='gray')
plt.show()
# 1.5 Filter to get edges
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                    [-10+0j, 0+ 0j, +10 +0j],
                    [ -3+3j, 0+10j,  +3 +3j]])

w1 = np.random.randn(3,3)
w2 = np.random.randn(3,3)
w3 = np.random.randn(3,3)

layer_1 = signal.convolve2d(current_img, w1, boundary='fill', mode='same')

layer_2 = signal.convolve2d(layer_1, w2, boundary='fill', mode='same')

layer_3 = signal.convolve2d(layer_2, w3, boundary='fill', mode='same')


fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(1, 3)
ax_orig.imshow(layer_1, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_mag.imshow(layer_2, cmap='gray')
ax_mag.set_title('Gradient magnitude')
ax_mag.set_axis_off()

ax_ang.imshow(layer_3, cmap='gray') # hsv is cyclic, like angles
ax_ang.set_title('Gradient orientation')
ax_ang.set_axis_off()
fig.show()
plt.show()

# ------- END OF THE CODE