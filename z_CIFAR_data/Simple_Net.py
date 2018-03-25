import os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from six.moves import cPickle as pickle
from read_10_data import get_data
from sklearn.utils import shuffle
import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(1)
np.random.seed(789)
tf.set_random_seed(789)

seq0 = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

# activation functions here and there
def tf_elu(x): return tf.nn.elu(x)
def d_tf_elu(x): return tf_elu(x) + 1.0 

def tf_softmax(x): return tf.nn.softmax(x)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(x): return tf.cast(tf.greater(x,0),dtype=tf.float32)

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_tanh(x): return tf.nn.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.square(tf_tanh(x))

def tf_atan(x): return tf.atan(x)
def d_tf_atan(x): return 1.0 / (1 + tf.square(x))

# === Get Data ===
train_images, train_labels, test_images,test_labels = get_data()

train_images_aug1 = seq.augment_images(train_images)
train_images_aug2 = seq.augment_images(train_images)

train_images1 = np.concatenate((train_images,train_images_aug1),axis=0)
train_images = np.concatenate((train_images1,train_images_aug2),axis=0)

train_labels1 = np.concatenate((train_labels,train_labels),axis=0)
train_labels = np.concatenate((train_labels,train_labels1),axis=0)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


# === Hyper ===
num_epoch =  10
batch_size = 100
print_size = 2
shuffle_size = 10
divide_size = 2

proportion_rate = 1000
decay_rate = 0.08
# decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)

beta1,beta2 = 0.9,0.999
adam_e = 0.00000001

# =========== Layer Class ===========
class CNNLayer():
      
  def __init__(self,kernel,in_c,out_c,act,d_act):
    with tf.device('/cpu:0'):
      self.w = tf.Variable(tf.truncated_normal([kernel,kernel,in_c,out_c],stddev=0.05,mean=0.0))
      self.act,self.d_act = act,d_act
      self.m,self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))

  def getw(self): return [self.w]
  def reg(self): return tf.nn.l2_loss(self.w)

  def feedforward(self,input,strids=1,padding="SAME"):
    self.input = input
    self.layer = tf.nn.conv2d(self.input,self.w,strides=[1,strids,strids,1],padding=padding)
    self.layerA = self.act(self.layer)
    return self.layerA

  def backprop(self,gradient):
    grad_part_1 = gradient
    grad_part_2 = self.d_act(self.layer)
    grad_part_3 = self.input

    grad_middle = tf.multiply(grad_part_1,grad_part_2)

    grad = tf.nn.conv2d_backprop_filter(
      input = grad_part_3,
      filter_sizes = self.w.shape,
      out_backprop = grad_middle,
      strides = [1,1,1,1],padding='SAME'
    )

    pass_size = list(self.input.shape[1:])
    grad_pass = tf.nn.conv2d_backprop_input(
      input_sizes =[batch_size]+pass_size,
      filter = self.w,
      out_backprop = grad_middle,
      strides = [1,1,1,1],padding='SAME'
    )

    grad_update = []
    grad_update.append(tf.assign(self.m,tf.add(beta1*self.m, (1-beta1)*grad)))
    grad_update.append(tf.assign(self.v,tf.add(beta2*self.v, (1-beta2)*grad**2)))

    m_hat = self.m / (1-beta1)
    v_hat = self.v / (1-beta2)
    adam_middel = learning_rate/(tf.sqrt(v_hat) + adam_e)
    grad_update.append(tf.assign(self.w,tf.subtract(self.w,tf.multiply(adam_middel,m_hat))))

    return grad_pass,grad_update
 
# === Make Layers ===
l1 = CNNLayer(3,3,96,tf_elu,d_tf_elu)
l2 = CNNLayer(3,96,96,tf_elu,d_tf_elu)
l3 = CNNLayer(3,96,96,tf_elu,d_tf_elu)

l4 = CNNLayer(3,96,192,tf_elu,d_tf_elu)
l5 = CNNLayer(3,192,192,tf_elu,d_tf_elu)
l6 = CNNLayer(3,192,192,tf_elu,d_tf_elu)

l7 = CNNLayer(3,192,192,tf_elu,d_tf_elu)
l8 = CNNLayer(1,192,192,tf_elu,d_tf_elu)
l9 = CNNLayer(1,192,10,tf_elu,d_tf_elu)

# === Make Graph ===
x = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
y = tf.placeholder(shape=[None,10],dtype=tf.float32)
learning_rate = tf.placeholder(shape=[],dtype=tf.float32)

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1)
layer3 = l3.feedforward(layer2,2)

layer4 = l4.feedforward(layer3)
layer5 = l5.feedforward(layer4)
layer6 = l6.feedforward(layer5,2)

layer7 = l7.feedforward(layer6)
layer8 = l8.feedforward(layer7,padding='VALID')
layer9 = l9.feedforward(layer8,padding='VALID')

global_avg_pool = tf.reduce_mean(layer9,[1,2])

final_soft = tf_softmax(global_avg_pool)
cost = tf.reduce_sum(-1.0 * (y*tf.log(final_soft) + (1.0-y)*tf.log(1.0-final_soft)))
correct_prediction = tf.equal(tf.argmax(final_soft, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- auto train ---
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# --- start the session ----
with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  train_total_cost,train_total_acc =0,0
  train_cost_overtime,train_acc_overtime = [],[]

  test_total_cost,test_total_acc = 0,0
  test_cost_overtime,test_acc_overtime = [],[]
  for iter in range(num_epoch):

        # Train Set
        for current_batch_index in range(0,int(len(train_images)/divide_size),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_batch_label = train_labels[current_batch_index:current_batch_index+batch_size,:]

            if iter >= 15  : 
              feed_dict = {x:current_batch,y:current_batch_label,learning_rate:0.000001}
            elif iter >= 13 :
              feed_dict = {x:current_batch,y:current_batch_label,learning_rate:0.000001}
            elif iter > 5 :
              feed_dict = {x:current_batch,y:current_batch_label,learning_rate:0.00001}
            else: 
              feed_dict = {x:current_batch,y:current_batch_label,learning_rate:0.0001}

            sess_results = sess.run([cost,accuracy,correct_prediction,auto_train],feed_dict=feed_dict)
            print("current iter:", iter,' Current batach : ',current_batch_index," current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
            train_total_cost = train_total_cost + sess_results[0]
            train_total_acc = train_total_acc + sess_results[1]

        # Test Set
        for current_batch_index in range(0,len(test_images),batch_size):
          current_batch = test_images[current_batch_index:current_batch_index+batch_size,:,:,:]
          current_batch_label = test_labels[current_batch_index:current_batch_index+batch_size,:]
          sess_results = sess.run( [cost,accuracy,correct_prediction], feed_dict= {x:current_batch,y:current_batch_label})
          print("\t\t\tTest Image Current iter:", iter,' Current batach : ',current_batch_index, " current cost: ", sess_results[0],' current acc: ',sess_results[1], end='\r')
          test_total_cost = test_total_cost + sess_results[0]
          test_total_acc = test_total_acc + sess_results[1]

        # store
        train_cost_overtime.append(train_total_cost/(len(train_images)/divide_size/batch_size ) ) 
        train_acc_overtime.append(train_total_acc /(len(train_images)/divide_size/batch_size )  )
        test_cost_overtime.append(test_total_cost/(len(test_images)/batch_size ) )
        test_acc_overtime.append(test_total_acc/(len(test_images)/batch_size ) )
        
        # print
        if iter%print_size == 0:
            print('\n=========')
            print("Avg Train Cost: ", train_cost_overtime[-1])
            print("Avg Train Acc: ", train_acc_overtime[-1])
            print("Avg Test Cost: ", test_cost_overtime[-1])
            print("Avg Test Acc: ", test_acc_overtime[-1])
            print('-----------')      

        if iter%shuffle_size ==  0: 
          print("\n==== shuffling iter: =====",iter," \n")
          train_images,train_labels = shuffle(train_images,train_labels)
          test_images,test_labels = shuffle(test_images,test_labels)
          
            
        train_total_cost,train_total_acc,test_total_cost,test_total_acc=0,0,0,0

  # plot and save
  plt.figure()
  plt.plot(range(len(train_cost_overtime)),train_cost_overtime,color='r')
  plt.title('Train Cost over time')
  plt.show()

  plt.figure()
  plt.plot(range(len(train_acc_overtime)),train_acc_overtime,color='b')
  plt.title('Train Acc over time')
  plt.show()

  plt.figure()
  plt.plot(range(len(test_cost_overtime)),test_cost_overtime,color='y')
  plt.title('Test Cost over time')
  plt.show()

  plt.figure()
  plt.plot(range(len(test_acc_overtime)),test_acc_overtime,color='g')
  plt.title('Test Acc over time')
  plt.show()
        








# -- end code --