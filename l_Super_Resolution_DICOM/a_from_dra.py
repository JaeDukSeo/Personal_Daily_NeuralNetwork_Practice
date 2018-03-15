import tensorflow as tf
import numpy as np
import os
from scipy import misc
np.random.seed(678)
tf.set_random_seed(678)

def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1 / (1 - tf.square(x))

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): tf_log(x) * (1- tf_log(X))

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLu(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)

# Make Class
class FCNN():
    
    def __init__(self,kenerl,inc,outc,act,d_act):
        self.w = tf.Variable(tf.random_normal([kenerl,kenerl,inc,outc]))
        self.act,self.d_act = act,d_act
        self.m,self.v = tf.Variable(tf.zeros_like(self.w)),tf.Variable(tf.zeros_like(self.w))
    def getw(self): return self.w
    def feedforward(self,input,stride_row,stride_col,padding,dilated=False,dilation=None):
        self.input  = input
        if dilated :
            self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride_row,stride_col,1],padding=padding,dilations=[1,dilation,dilation,1])
        else:
            self.layer  = tf.nn.conv2d(input,self.w,strides=[1,stride_row,stride_col,1],padding=padding)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self,gradient=None,og_input=None):
        # grad_part_1 = tf.multiply(gradient, og_input)
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.input

        grad_middle = tf.transpose(tf.transpose(tf.multiply(grad_part_1,grad_part_2),[0,2,1,3]),[0,2,1,3])

        grad = tf.nn.conv2d_backprop_filter(input=grad_part_3,filter_sizes=self.w.shape,
        out_backprop=grad_middle,strides=[1,1,1,1],padding='SAME')

        pass_size = list(self.input.shape[1:])
        pass_on_grad = tf.nn.conv2d_backprop_input(input_sizes=[batch_size]+pass_size,filter=self.w,
        out_backprop=grad_middle,strides=[1,1,1,1],padding='SAME')

        grad_update = []
        grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
        grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

        m_hat = self.m / (1-beta1)
        v_hat = self.v / (1-beta2)
        adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
        grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

        return pass_on_grad,grad_update

# 0. Get the list
PathDicom = "../lung_data_1/"
lstFilesDCM1 = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM1.append(os.path.join(dirName,filename))

# 1. Read the data into Numpy
one = np.zeros((119,512,512,1))
print(one.sum())

# 1.5 Transfer All of the Data into array
print('===== READING DATA ========')
for file_index in range(len(lstFilesDCM1)):
    one[file_index,:,:]   = np.expand_dims(misc.imread(lstFilesDCM1[file_index],mode='F').astype(np.float32),axis=3)
print('===== Done READING DATA ========')

print(one.sum())



temp = np.ones((1,156,156,1)).astype(np.float32)
# Make Layer Object
l1 = FCNN(7,1,3,tf_ReLU,d_tf_ReLu)
l2 = FCNN(5,3,5,tf_ReLU,d_tf_ReLu)
l3 = FCNN(3,5,7,tf_ReLU,d_tf_ReLu)
l4 = FCNN(2,7,5,tf_ReLU,d_tf_ReLu)
l5 = FCNN(1,5,1,tf_ReLU,d_tf_ReLu)

# Make graph
x = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,512,512,1],dtype=tf.float32)

with tf.Session() as sess:

    layer1 = l1.feedforward(temp,1,1,"SAME")
    layer2 = l2.feedforward(layer1,1,1,"SAME")
    layer3 = l3.feedforward(layer2,1,1,"SAME")
    layer4 = l4.feedforward(layer3,2,2,"SAME")
    layer5 = l5.feedforward(layer4,1,1,"SAME")
    
    print(layer1.shape)
    print(layer2.shape)
    print(layer3.shape)
    print(layer4.shape)
    print(layer5.shape)
    




# -- end code --