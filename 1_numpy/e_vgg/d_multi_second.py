import numpy as np,sys
from sklearn.datasets import load_digits
from scipy.ndimage.filters import maximum_filter
import skimage.measure
from scipy.signal import convolve2d
from scipy import fftpack

def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))

def fftdeconvolve(in1, in2, mode="full"):
    """Deconvolve two N-dimensional arrays using FFT. See convolve.

    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))
    size = s1+s2-1

    # Always use 2**n-sized FFT
    fsize = 2**np.ceil(np.log2(size))
    IN1 = fftpack.fftn(in1,fsize)
    IN1 /= fftpack.fftn(in2,fsize)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = fftpack.ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.product(s1,axis=0) > np.product(s2,axis=0):
            osize = s1
        else:
            osize = s2
        return _centered(ret,osize)
    elif mode == "valid":
        return _centered(ret,abs(s2-s1)+1)

np.random.seed(12314)

def ReLU(x):
    mask  = (x >0) * 1.0 
    return mask * x
def d_ReLU(x):
    mask  = (x >0) * 1.0 
    return mask 

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def arctanh(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / ( 1 + x ** 2)

def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))

# 1. Prepare Data
data =load_digits()
image = data.images
label = data.target
num_epoch = 1
learning_rate = 0.0001
total_error = 0

# 1. Prepare only one and only zero
only_zero_index = np.asarray(np.where(label == 0))
only_one_index  = np.asarray(np.where(label == 1))

# 1.5 prepare Label
only_zero_label = label[only_zero_index].T
only_one_label  = label[only_one_index].T
image_label = np.vstack((only_zero_label,only_one_label))

# 2. prepare matrix image
only_zero_image = np.squeeze(image[only_zero_index])
only_one_image = np.squeeze(image[only_one_index])
image_matrix = np.vstack((only_zero_image,only_one_image))

# 4. Declare hyper parameter
w1a = np.random.randn(3,3)
w1b = np.random.randn(3,3)

w2a = np.random.randn(3,3)
w2b = np.random.randn(3,3)
w2c = np.random.randn(3,3)
w2d = np.random.randn(3,3)

w3 = np.random.randn(16,28)
w4 = np.random.randn(28,36)
w5 = np.random.randn(36,1)

    
for image_index in range(len(image_matrix)):

    current_image  = image_matrix[image_index]
    current_image_label  = image_label[image_index]

    l1aIN = np.pad(current_image,1,mode='constant')

    l1a = convolve2d(l1aIN,w1a,mode='valid')
    l1aA = ReLU(l1a)
    l1aM = skimage.measure.block_reduce(l1aA, block_size=(2,2), func=np.max)

    l1b = convolve2d(l1aIN,w1b,mode='valid')
    l1bA = arctanh(l1b)
    l1bM = skimage.measure.block_reduce(l1bA, block_size=(2,2), func=np.max)
    
    la2INa = np.pad(l1aM,1,mode='constant')
    la2INb = np.pad(l1bM,1,mode='constant')

    l2a = convolve2d(la2INa,w2a,mode='valid')
    l2aA = ReLU(l2a)
    l2aM = skimage.measure.block_reduce(l2aA, block_size=(2,2), func=np.max)

    l2b = convolve2d(la2INa,w2b,mode='valid')
    l2bA = tanh(l2b)
    l2bM = skimage.measure.block_reduce(l2bA, block_size=(2,2), func=np.max)

    l2c = convolve2d(la2INb,w2c,mode='valid')
    l2cA = ReLU(l2c)
    l2cM = skimage.measure.block_reduce(l2cA, block_size=(2,2), func=np.max)

    l2d = convolve2d(la2INb,w2d,mode='valid')
    l2dA = arctanh(l2d)
    l2dM = skimage.measure.block_reduce(l2dA, block_size=(2,2), func=np.max)  

    l3IN = np.expand_dims(np.hstack((l2aM.ravel(),l2bM.ravel(),l2cM.ravel(),l2dM.ravel())),axis=0)
    l3 = l3IN.dot(w3)
    l3A = arctanh(l3)

    l4 = l3A.dot(w4)
    l4A = ReLU(l4)
    
    l5 = l4A.dot(w5)
    l5A = log(l5)

    cost = np.square(l5A - current_image_label).sum() * 0.5
    total_error = total_error + cost

    grad_5_part_1 = l5A - current_image_label
    grad_5_part_2 = d_log(l5)
    grad_5_part_3 = l4A
    grad_5 =    grad_5_part_3.T.dot(grad_5_part_1 * grad_5_part_2)     
    
    grad_4_part_1 = (grad_5_part_1 * grad_5_part_2).dot(w5.T)
    grad_4_part_2 = d_ReLU(l4)
    grad_4_part_3 = l3A
    grad_4 =    grad_4_part_3.T.dot(grad_4_part_1 * grad_4_part_2)   

    grad_3_part_1 = (grad_4_part_1 * grad_4_part_2).dot(w4.T)
    grad_3_part_2 = d_arctan(l3)
    grad_3_part_3 = l3IN
    grad_3 =     grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

    grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)

    grad_2_part_1_a = np.reshape(grad_2_part_1[:,:4],(2,2))
    grad_2_part_1_b = np.reshape(grad_2_part_1[:,4:8],(2,2))
    grad_2_part_1_c = np.reshape(grad_2_part_1[:,8:12],(2,2))
    grad_2_part_1_d = np.reshape(grad_2_part_1[:,12:],(2,2))

    grad_2_mask_a =  np.equal(l2aA, l2aM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
    grad_2_mask_b =  np.equal(l2bA, l2bM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
    grad_2_mask_c =  np.equal(l2cA, l2cM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 
    grad_2_mask_d =  np.equal(l2dA, l2dM.repeat(2, axis=0).repeat(2, axis=1)).astype(int) 

    grad_2_winodw_a =  grad_2_mask_a *  grad_2_part_1_a.repeat(2, axis=0).repeat(2, axis=1) 
    grad_2_winodw_b =  grad_2_mask_b *  grad_2_part_1_b.repeat(2, axis=0).repeat(2, axis=1) 
    grad_2_winodw_c =  grad_2_mask_c *  grad_2_part_1_c.repeat(2, axis=0).repeat(2, axis=1) 
    grad_2_winodw_d =  grad_2_mask_d *  grad_2_part_1_d.repeat(2, axis=0).repeat(2, axis=1) 
    
    grad_2_part_2_a,grad_2_part_2_b = d_ReLU(l2a),d_tanh(l2b)
    grad_2_part_2_c,grad_2_part_2_d = d_ReLU(l2c),d_arctan(l2d)
    
    grad_2_part_3_a,grad_2_part_3_b = la2INa,la2INa
    grad_2_part_3_c,grad_2_part_3_d = la2INb,la2INb

    grad_2a = convolve2d(grad_2_part_3_a,np.rot90(grad_2_winodw_a *grad_2_part_2_a,2 ),mode='valid')
    grad_2b = convolve2d(grad_2_part_3_b,np.rot90(grad_2_winodw_b *grad_2_part_2_b,2 ),mode='valid')
    grad_2c = convolve2d(grad_2_part_3_c,np.rot90(grad_2_winodw_c *grad_2_part_2_c,2 ),mode='valid')
    grad_2d = convolve2d(grad_2_part_3_d,np.rot90(grad_2_winodw_d *grad_2_part_2_d,2 ),mode='valid')

    temp = np.pad(grad_2_winodw_a *grad_2_part_2_a,1,mode='constant')
    print(temp.shape)
    temp = convolve2d(temp,w2a,mode='valid')
    print(temp.shape)

    
    
    
    # deconvolved = restoration.richardson_lucy(grad_3_part_in_a, w2a)    


    sys.exit()


# -- end code ---