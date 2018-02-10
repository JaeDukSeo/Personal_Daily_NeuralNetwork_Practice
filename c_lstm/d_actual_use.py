from PIL import Image
import numpy as np,os,sys,cv2
from matplotlib import pyplot as plt
from scipy.signal import convolve2d,convolve

np.random.seed(5678)
np.set_printoptions(precision=3,suppress=True)


def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2
def ReLu(x):
    mask = (x>0) * 1.0
    return mask *x
def d_ReLu(x):
    mask = (x>0) * 1.0
    return mask 
def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))
def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1 + x ** 2)









# 1. Preprocess the data
Pathgif = "./data/dog_preprocessed/"
dog_gif = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Pathgif):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            dog_gif.append(os.path.join(dirName,filename))

Pathgif = "./data/baby_preprocessed/"
baby_gif = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Pathgif):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            baby_gif.append(os.path.join(dirName,filename))

dog_array = np.zeros((24,100,100,3))
store_index = 0
for element in dog_gif:
    img = Image.open(element) 
    for iter in range(img.n_frames):
        img.seek(iter)
        new_frame = np.array(img.convert('RGB'))
        dog_array[store_index,:,:,:] = new_frame
        store_index = store_index + 1

baby_array = np.zeros((24,100,100,3))
store_index = 0
for element in baby_gif:
    img = Image.open(element) 
    for iter in range(img.n_frames):
        img.seek(iter)
        new_frame = np.array(img.convert('RGB'))
        baby_array[store_index,:,:,:] = new_frame
        store_index = store_index + 1

dog_label = np.ones((24,1))
baby_label = np.zeros((24,1))
train_num = 18

train_data = np.vstack((dog_array[:train_num,:,:,:],baby_array[:train_num,:,:,:]))
train_label= np.vstack((dog_label[:train_num,:],baby_label[:train_num,:]))
test_data  = np.vstack((dog_array[train_num:,:,:,:],baby_array[train_num:,:,:,:]))
test_label = np.vstack((dog_label[train_num:,:],baby_label[train_num:,:]))

# 2. Declare the weights
num_epoch = 1
wf = np.random.randn(3,3)
wf = np.zeros((3,3))
wi = np.random.randn(3,3)
wo = np.random.randn(3,3)
wrec = np.random.randn(3,3)

hidden_state = np.random.randn(4,102,102,3)

# # 3. Create the loop
for iter in range(num_epoch):
    
    for image_index in range(0,len(train_data),3):
    
        current_first_input = train_data[image_index,:,:,:]

        l1Input = np.pad(current_first_input,((1,1),(1,1),(0,0)),'constant')
        l1f = convolve(l1Input,np.expand_dims(wf,axis=2),mode='valid') + \
              convolve(hidden_state[0,:,:,:],np.expand_dims(wrec,axis=2),'valid')
        l1fA = log(l1f)
        print(l1fA.shape)

        # l1f = convolve(current_image,np.expand_dims(wf,axis=2),mode='valid') 





# -- end code --