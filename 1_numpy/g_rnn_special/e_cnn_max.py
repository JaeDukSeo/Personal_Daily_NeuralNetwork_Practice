import numpy as np
import scipy
from scipy import signal

np.random.seed(56789)


image = np.random.randn(4,4)
scharr = np.random.randn(3,3)

grad = signal.convolve2d(image, scharr, boundary='fill',fillvalue=0, mode='same')
print(grad.shape)

w2 = np.random.randn(1,16)
out = w2.dot(np.reshape(grad,(16,1)))

print(out.shape)


grad_2_part_1 = (out - 9)
grad_2 = grad_2_part_1 * np.reshape(grad,(16,1))

grad_1_part_1 = grad_2_part_1
grad_1_part_2 = w2
grad_1_part_3 = image

grad_1_reshape = np.reshape((grad_1_part_1*grad_1_part_2),(4,4))
print(grad_1_reshape.shape)
# image = np.random.randn(6,6)
ss =np.rot90(grad_1_reshape,2)
print(ss.shape)
sss = np.pad(image,(1,1),'constant', constant_values=(0,0))
print(sss.shape)

safasdafsa = signal.convolve2d(sss, ss,mode='valid')
print('-------')
print(safasdafsa.shape)

