import tensorflow as tf
import numpy as np,sys
from  scipy.signal import convolve2d

np.random.seed(678)
tf.set_random_seed(6789)
sess = tf.Session()

# ========== Experiment Set Up ===========
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
print("====== Original Set Up ======")
print("Matrix Shape : ",matrix.shape)
print(matrix)
print("kernel Shape : ",kernel.shape)
print(kernel)
# ========== Experiment Set Up ===========

# ========== EXAMPLE 1 - Dilation Factor 1 ===========
print("\n====== Dilated Kernel 1 ======")
print('Kernal For "Familiar" Convolution for Numpy: \n',kernel)

print("========== Numpy 'familiar' Convolution Results ===============")
np_results = convolve2d(matrix,kernel,mode='valid')
print("Numpy Results Shape: ",np_results.shape)
print(np_results)

print("========== Tensorfow Conv2D Results ===============")
tf_opreation1_1 = tf.nn.conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                            np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                            strides=[1,1,1,1],padding="VALID",
                            dilations=[1,1,1,1])
tf_result = sess.run(tf_opreation1_1)
print("Tensorfow Conv2D Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))

print("========== Tensorfow Atrous Conv2D Results ===============")
tf_opreation1_2 = tf.nn.atrous_conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                                    np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                                    rate=1,padding="VALID")
tf_result = sess.run(tf_opreation1_2)
print("Tensorfow Atrous Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))
# ========== EXAMPLE 1 - Dilation Factor 1 ===========




# ========== EXAMPLE 2 - Dilation Factor 2 ===========
print("\n====== Dilated Kernel 2 ======")
kernel2 = np.array([
    [1,0,1,0,1],
    [0,0,0,0,0],
    [1,0,1,0,1],
    [0,0,0,0,0],
    [1,0,1,0,1]
])
print('Kernal For "Familiar" Convolution for Numpy: \n',kernel2)

print("========== Numpy 'familiar' Convolution Results ===============")
np_results = convolve2d(matrix,kernel2,mode='valid')
print("Numpy Results Shape: ",np_results.shape)
print(np_results)

print("========== Tensorfow Conv2D Results ===============")
tf_opreation2_1 = tf.nn.conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                            np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                            strides=[1,1,1,1],padding="VALID",
                            dilations=[1,2,2,1])
tf_result = sess.run(tf_opreation2_1)
print("Tensorfow Conv2D Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))

print("========== Tensorfow Atrous Conv2D Results ===============")
tf_opreation2_2 = tf.nn.atrous_conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                                    np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                                    rate=2,padding="VALID")
tf_result = sess.run(tf_opreation2_2)
print("Tensorfow Atrous Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))
# ========== EXAMPLE 2 - Dilation Factor 2 ===========




# ========== EXAMPLE 3 - Dilation Factor 3 ===========
print("\n====== Dilated Kernel 3 ======")
kernel2 = np.array([
    [1,0,0,1,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,1,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,1,0,0,1]
])
print('Kernal For "Familiar" Convolution for Numpy: \n',kernel2)

print("========== Numpy 'familiar' Convolution Results ===============")
np_results = convolve2d(matrix,kernel2,mode='valid')
print("Numpy Results Shape: ",np_results.shape)
print(np_results)

print("========== Tensorfow Conv2D Results ===============")
tf_opreation3_1 = tf.nn.conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                            np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                            strides=[1,1,1,1],padding="VALID",
                            dilations=[1,3,3,1])
tf_result = sess.run(tf_opreation3_1)
print("Tensorfow Conv2D Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))

print("========== Tensorfow Atrous Conv2D Results ===============")
tf_opreation3_2 = tf.nn.atrous_conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                                    np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                                    rate=3,padding="VALID")
tf_result = sess.run(tf_opreation3_2)
print("Tensorfow Atrous Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))
# ========== EXAMPLE 3 - Dilation Factor 3 ===========


sys.exit()




# ========== EXAMPLE 4 - Dilation Factor 4 ===========
print("\n====== Dilated Kernel 4 ======")
kernel2 = np.array([
    [1,0,0,1,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,1,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,1,0,0,1]
])
print('Kernal For "Familiar" Convolution for Numpy: \n',kernel2)

print("========== Numpy 'familiar' Convolution Results ===============")
np_results = convolve2d(matrix,kernel2,mode='valid')
print("Numpy Results Shape: ",np_results.shape)
print(np_results)

print("========== Tensorfow Conv2D Results ===============")
tf_opreation3_1 = tf.nn.conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                            np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                            strides=[1,1,1,1],padding="VALID",
                            dilations=[1,3,3,1])
tf_result = sess.run(tf_opreation3_1)
print("Tensorfow Conv2D Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))

print("========== Tensorfow Atrous Conv2D Results ===============")
tf_opreation3_2 = tf.nn.atrous_conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                                    np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                                    rate=3,padding="VALID")
tf_result = sess.run(tf_opreation3_2)
print("Tensorfow Atrous Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))
# ========== EXAMPLE 4 - Dilation Factor 4 ===========
sys.exit()


# 2. Numpy 2D convolution
print("========== NP Results ===============")
np_results = convolve2d(matrix,kernel,mode='valid')
print(np_results.shape)
print(np_results)


# 3. TF convolution
print("========== Tensorfow Conv2D Results ===============")
tf_operation1 = tf.nn.conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                            np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                            strides=[1,1,1,1],padding="VALID")
tf_result = sess.run(tf_operation1)
print(tf_result.shape)
print(np.squeeze(tf_result))


# 4. NP Dilated Convolution of 2
kernel2 = np.array([
    [1,0,1,0,1],
    [0,0,0,0,0],    
    [1,0,1,0,1],
    [0,0,0,0,0],    
    [1,0,1,0,1],
]).astype(np.float32) 
print("========== NP Results Dilated 2 ===============")
np_results = convolve2d(matrix,kernel2,mode='valid')
print(np_results.shape)
print(np_results)


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


print("========== Tensorfow Results Dilated Value of 2 ===============")
tf_opreation4 = tf.nn.conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                            np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                            strides=[1,1,1,1],padding="VALID",
                            dilations=[1,2,2,1])
tf_result = sess.run(tf_opreation4)
print(tf_result.shape)
print(np.squeeze(tf_result))


sys.exit()


# 7. TF Dilated with Dilation of 3
print("========== Tensorfow Results Dilated Value of 3 ===============")
tf_opreation4 = tf.nn.atrous_conv2d(np.expand_dims(np.expand_dims(matrix,axis=0),axis=3),
                                    np.expand_dims(np.expand_dims(kernel,axis=3),axis=4),
                                    rate=3,padding="VALID")
tf_result = sess.run(tf_opreation4)
print(tf_result.shape)
print(np.squeeze(tf_result))


# -- end code --