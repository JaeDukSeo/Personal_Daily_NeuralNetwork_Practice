import tensorflow as tf
import numpy as np

np.random.seed(678)
tf.set_random_seed(678)
# np.set_printoptions(precision=4)
sess = tf.InteractiveSession()


def batchnorm_forward(x, gamma, beta, eps):

  N,_,_,_ = x.shape

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

# create the test data and expand dim
test_data = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
]).astype(np.float32)
test_data = np.expand_dims(test_data,axis=0)
test_data = np.expand_dims(test_data,axis=3)

for _ in range(3):
    test_data = np.vstack((test_data,test_data))

print(test_data.shape)
print(test_data.mean())
print(test_data.max())
print(test_data.min())

tf_batch_norm = tf.nn.batch_normalization(test_data,mean=np.mean(test_data),variance=np.var(test_data),scale=1.0,offset=0.0,variance_epsilon=1e8).eval()
print(tf_batch_norm.shape)
print(tf_batch_norm.mean())
print(tf_batch_norm.max())
print(tf_batch_norm.min())

np_batch_norm,_ = batchnorm_forward(test_data,1.0,0,1e8)
print(np_batch_norm.shape)
print(np.finfo(np_batch_norm.mean()).precision   )
print(np.finfo(np_batch_norm.max()).precision   )
print(np.finfo(np_batch_norm.min()).precision   )



# -- end code --