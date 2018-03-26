import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(678)
tf.set_random_seed(678)
# np.set_printoptions(precision=4)
sess = tf.InteractiveSession()



# create the test data and expand dim
test_data = np.random.poisson(5,10000)

print(test_data.shape)
print(test_data.mean())
print(test_data.max())
print(test_data.min())
plt.hist(test_data, bins=30)
plt.show()


tf_batch_norm = tf.nn.batch_normalization(test_data,mean=np.mean(test_data),variance=np.var(test_data),scale=1.0,offset=0.0,variance_epsilon=1e8).eval()
print(tf_batch_norm.shape)
print(tf_batch_norm.mean())
print(tf_batch_norm.max())
print(tf_batch_norm.min())
plt.hist(tf_batch_norm, bins=30)
plt.show()




# -- end code --