import numpy as np
import tensorflow as tf
import sklearn 
import matplotlib.pyplot as plt
import sys

np.random.seed(6789)

# create random data
data = np.random.normal(size=[10,5])

alpa,beta = 1.0,1.0
batch_e = 0.00001

data_mean = np.sum(data)/len(data)
print(data_mean.shape)
mini_var = np.sum(np.square(data-data_mean)) / len(data)
print(mini_var.shape)
normalize = (data-data_mean)/(np.sqrt(mini_var) + batch_e)
print(normalize.shape)

output = alpa*normalize + beta

# print(data)
# print("MAx: ",data.max())
# print("Min: ",data.min())
# print("Meanx: ",data.mean())

# print('========')
# print(normalize)
# print("MAx: ",normalize.max())
# print("Min: ",normalize.min())
# print("Meanx: ",normalize.mean())

# print('========')
# print(output)
# print("MAx: ",output.max())
# print("Min: ",output.min())
# print("Meanx: ",output.mean())

def batchnorm_forward(x, gamma, beta, eps):
    
  N, D = x.shape

  #step1: calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache


print('--------')
data_mean = np.sum(data,axis=0)/len(data)
print(data_mean.shape)
mini_var = np.sum(np.square(data-data_mean),axis=0) / len(data)
print(mini_var.shape)
normalize = (data-data_mean)/(np.sqrt(mini_var) + batch_e)
print(normalize.shape)
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
print('========')
print('========')


sss = batchnorm_forward(data,1.0,1.0,batch_e)
print(sss[0])

print('========')
print('========')

print(( np.round(sss[0],decimals=4)- np.round(output,decimals=4) ).sum())

# -- end code --