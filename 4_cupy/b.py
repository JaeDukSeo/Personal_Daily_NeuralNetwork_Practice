import urllib.request
from io import BytesIO
from PIL import Image
import numpy as np
import cupy as cp,time,sys
import matplotlib.pyplot as plt

from numba import jit, double

s = 15
array_a = np.random.rand(s ** 3).reshape(s, s, s)
array_b = np.random.rand(s ** 3).reshape(s, s, s)

# Original code
def custom_convolution(A, B):

    dimA = A.shape[0]
    dimB = B.shape[0]
    dimC = dimA + dimB

    C = np.zeros((dimC, dimC, dimC))
    for x1 in range(dimA):
        for x2 in range(dimB):
            for y1 in range(dimA):
                for y2 in range(dimB):
                    for z1 in range(dimA):
                        for z2 in range(dimB):
                            x = x1 + x2
                            y = y1 + y2
                            z = z1 + z2
                            C[x, y, z] += A[x1, y1, z1] * B[x2, y2, z2]
    return C

# Numba'ing the function with the JIT compiler
fast_convolution = jit(double[:, :, :](double[:, :, :],
                        double[:, :, :]))(custom_convolution)

def download_very_big_image():
    url = 'http://i.imgur.com/DEKdmba.jpg'
    conn = urllib.request.urlopen(url)
    file = np.array(Image.open(BytesIO(conn.read())))
    return file
    
image = download_very_big_image()[:,:,1]
from scipy import signal
print(image.shape)
# plt.imshow(image)
# plt.show()

def convolve2dcp(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    kernel = cp.flipud(cp.fliplr(kernel))    # Flip the kernel
    output = cp.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = cp.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    return output
def convolve2d(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    return output

image =np.random.randn(6000,6000)
cp_image = cp.asarray(image)



start = time.time()
w22 = np.random.randn(100,100)
grad = signal.convolve(image, w22,  mode='same')
end = time.time()
print(end - start,"\n-------------")


start = time.time()
blurred = signal.fftconvolve(image, w22, mode='same')
end = time.time()
print(end - start,"\n-------------")


print((grad - blurred).sum())

sys.exit()
face = misc.face(gray=True)
kernel = np.outer(signal.gaussian(70, 8), signal.gaussian(70, 8))
blurred = signal.fftconvolve(face, kernel, mode='same')





start = time.time()
image = np.random.randn(1413,2120)
w22 = np.random.randn(100,100)

fast_result = fast_convolution(image, w22)
end = time.time()
print(end - start,"\n-------------")

# --- end code --