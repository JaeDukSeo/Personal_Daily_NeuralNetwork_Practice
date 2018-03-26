import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(678)
tf.set_random_seed(678)
# np.set_printoptions(precision=4)
sess = tf.InteractiveSession()

# create the test data and expand dim
test_data =50*np.random.rand() * np.random.normal(10, 10, 100) + 20

print(np.mean(test_data))
print(np.var(test_data))
print(np.max(test_data))
print(np.min(test_data))
plt.hist(test_data,bins='auto')
plt.show()

tf_batch_norm = tf.nn.batch_normalization(test_data,mean=np.mean(test_data),variance=np.var(test_data),scale=1.0,offset=0.0,variance_epsilon=1e8).eval()
print(np.mean(tf_batch_norm))
print(np.var(tf_batch_norm))
print(np.max(tf_batch_norm))
print(np.min(tf_batch_norm))
plt.hist(tf_batch_norm, bins='auto')
plt.show()




# -- end code --