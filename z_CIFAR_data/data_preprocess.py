from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
from pylearn2.datasets.cifar10 import CIFAR10
import textwrap
import sys

# load data and GCN them
train = CIFAR10(which_set='train', gcn=55.)

# apply zca
# preprocessor = preprocessing.ZCA()
# train.apply_preprocessor(preprocessor=preprocessor, can_fit=True)

# save data
train.use_design_loc('./cifar10_pkl/train.npy')
serial.save('./cifar10_pkl/train.pkl', train)

# load test
test = CIFAR10(which_set='test', gcn=55.)

# apply zca
# test.apply_preprocessor(preprocessor=preprocessor, can_fit=False)

# save data
test.use_design_loc('./cifar10_pkl/test.npy')
serial.save('./cifar10_pkl/test.pkl', test)

# save preprocessor
# serial.save('./cifar10_preprocessed/preprocessor.pkl', preprocessor)



# -- end code --