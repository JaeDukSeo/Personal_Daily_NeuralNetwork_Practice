import numpy as np
import tensorflow as tf
import sklearn 
import matplotlib.pyplot as plt

np.random.seed(6789)

# create random data
data = np.random.normal(size=[10,5])

alpa,beta = 1.0,1.0
batch_e = 0.00001

data_mean = np.sum(data)/len(data)
mini_var = np.sum(np.square(data-data_mean)) / len(data)
normalize = (data-data_mean)/(np.sqrt(mini_var) + batch_e)

output = alpa*normalize + beta

print(data)
print("MAx: ",data.max())
print("Min: ",data.min())
print("Meanx: ",data.mean())

print('========')
print(normalize)
print("MAx: ",normalize.max())
print("Min: ",normalize.min())
print("Meanx: ",normalize.mean())

print('========')
print(output)
print("MAx: ",output.max())
print("Min: ",output.min())
print("Meanx: ",output.mean())

# -- end code --