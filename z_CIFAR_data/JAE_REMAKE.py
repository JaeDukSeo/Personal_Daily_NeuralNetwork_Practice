import os,sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from read_10_data import get_data
from sklearn.utils import shuffle
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)
np.random.seed(789)
tf.set_random_seed(789)

# To speed up the training
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
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

# === Augment Data ===
# train_images_augmented = seq.augment_images(train_images)
# train_images = np.concatenate((train_images,train_images_augmented),axis=0)
# train_labels = np.concatenate((train_labels,train_labels),axis=0)
# train_images,train_labels = shuffle(train_images,train_labels)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

















# -- end code --