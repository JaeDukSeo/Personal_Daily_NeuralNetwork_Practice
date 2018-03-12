import tensorflow as tf
import numpy as np
from  scipy.signal import convolve2d

np.random.seed(678)
tf.set_random_seed(6789)
sess = tf.Session()

# 0. Create a matrix we want to perform experiments
mat_size = 10
matrix = np.zeros((mat_size,mat_size)).astype(np.float32) 

for x in range(4,7):
    for y in range(3,6):
        matrix[y,x] = 1

# 1. Create a Kernel 
kernel = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
]).astype(np.float32) 
print(matrix.shape)
print(matrix)

# 2. Numpy 2D convolution
print("========== NP Results ===============")
np_results = convolve2d(matrix,kernel,mode='valid')
print(np_results.shape)
print(np_results)


# 3. TF convolution
print("========== Tensorfow Results ===============")
tf_operation1 = tf.nn.conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                            np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                            strides=[1,1,1,1],padding="VALID")
tf_result = sess.run(tf_operation1)
print(tf_result.shape)
print(np.squeeze(tf_result))


# 4. 

# 5. TF Dilated with Dilation of 1
print("========== Tensorfow Results Dilated Value of 1 ===============")
tf_opreation2 = tf.nn.atrous_conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                                    np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                                    rate=1,padding="VALID")
tf_result = sess.run(tf_opreation2)
print(tf_result.shape)
print(np.squeeze(tf_result))

# 6. TF Dilated with Dilation of 2
print("========== Tensorfow Results Dilated Value of 2 ===============")
tf_opreation3 = tf.nn.atrous_conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                                    np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                                    rate=2,padding="VALID")
tf_result = sess.run(tf_opreation3)
print(tf_result.shape)
print(np.squeeze(tf_result))

# 7. TF Dilated with Dilation of 3
print("========== Tensorfow Results Dilated Value of 3 ===============")
tf_opreation4 = tf.nn.atrous_conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                                    np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                                    rate=3,padding="VALID")
tf_result = sess.run(tf_opreation4)
print(tf_result.shape)
print(np.squeeze(tf_result))


# -- end code --