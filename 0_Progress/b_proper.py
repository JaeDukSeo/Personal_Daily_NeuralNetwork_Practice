import tensorflow as tf
import numpy as np
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
        
        self.w1 = tf.Variable(tf.random_normal([893,4],seed=0))

    def feed_forward(self,input=None):
        
        return 8



class discriminator():
    
    def __init__(self):
        print(7)

G = generator



# -- end code --