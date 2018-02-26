import tensorflow as tf
import numpy as np,os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.ndimage import imread

# -2. Set the Random Seed Values
tf.set_random_seed(789)
np.random.seed(568)

# -1 Tf activation functions

# 0. Get the list
PathDicom = "../lung_data_1/"
lstFilesDCM1 = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(PathDicom)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM1.append(os.path.join(dirName,filename))

# 1. Read the data into Numpy
one = np.zeros((119,512,512))
two = np.zeros((119,512,512))
three = np.zeros((119,512,512))

# 1.5 Transfer All of the Data into array
print('===== READING DATA ========')
for file_index in range(len(lstFilesDCM1)):
    one[file_index,:,:]   = imread(lstFilesDCM1[file_index],mode='F')
    # two[file_index,:,:]   = imread(lstFilesDCM2[file_index],mode='F')
    # three[file_index,:,:]   = imread(lstFilesDCM3[file_index],mode='F')
print('===== Done READING DATA ========')

training_data = one
# training_data = np.vstack((one,two,three))

# 2. Create the Class
class generator():
    
    def __init__(self):
        
        w1 = tf.Variable(tf.random_normal([893,4],seed=0))




input = tf.expand_dims(tf.expand_dims(tf.Variable(training_data[0,:,:],dtype=tf.float32),axis=0),axis=3)
filter = tf.Variable(tf.random_normal([5,5,1,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
op2 = tf.nn.conv2d(op, filter, strides=[1, 1, 1, 1], padding='SAME')
op3 = tf.nn.conv2d(op2, filter, strides=[1, 1, 1, 1], padding='SAME')
op4 = tf.nn.conv2d(op3, filter, strides=[1, 1, 1, 1], padding='SAME')

# input = tf.Variable(tf.random_normal([1,3,3,5]))
# filter = tf.Variable(tf.random_normal([1,1,5,1]))

# op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    print("filter")
    print(filter.eval())
    print("result")
    result = sess.run(op4)

    print(result.shape)
    plt.imshow(np.squeeze(result),cmap='gray')
    plt.show()

# -- end code ---