import csv
import numpy as np
from numpy import genfromtxt
import pandas as pd
np.set_printoptions(2,suppress =True,nanstr="Non")
my_data = genfromtxt('depression.csv', delimiter=',')

print(my_data.shape)


df = pd.read_csv('depression.csv')


print(df)
# -- end code --