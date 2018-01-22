import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digit = load_digits()

image = digit.images
label = digit.target


only_zero_index = np.asarray(np.where(label == 0))
only_one_index  = np.asarray(np.where(label == 1))


only_zero_image = np.squeeze(image[only_zero_index])
only_one_image = np.squeeze(image[only_one_index])

for iter in range(20):
    
    plt.imshow(only_zero_image[iter,:,:],cmap='gray')
    plt.pause(0.5)

for iter in range(20):
    
    plt.imshow(only_one_image[iter,:,:],cmap='gray')
    plt.pause(0.5)


plt.show()

# -- end code --