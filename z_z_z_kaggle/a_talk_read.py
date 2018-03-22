import numpy as np
import tensorflow as tf
from numpy import genfromtxt
import zipfile
import pandas as pd

# read all of the data
# zf = zipfile.ZipFile('a_talking/train.csv.zip') 
# full_data = genfromtxt(zf.open('mnt/ssd/kaggle-talkingdata2/competition_files/train.csv'),delimiter=',')

# read sample data
zf = zipfile.ZipFile('a_talking/train_sample.csv.zip') 
# full_data = genfromtxt(zf.open('mnt/ssd/kaggle-talkingdata2/competition_files/train_sample.csv'),delimiter=',')
full_data = genfromtxt('a_talking/train_sample.csv',delimiter=',')

# read as data frame
full_data = pd.read_csv('a_talking/train_sample.csv', nrows=227*2, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print( full_data.groupby(full_data['is_attributed'])['is_attributed'].count())



# -- end code --