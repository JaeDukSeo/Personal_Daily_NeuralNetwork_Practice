import tensorflow as tf
import numpy as np

temp = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
]).astype(np.float32)

print(temp.repeat(2,axis=0).repeat(2,axis=1))


x = [[1,2],[3,4]]
print(np.repeat(3, 4))
print(np.repeat(x, 2))
print(np.repeat(x, 3, axis=1))
print("=============")

x = tf.constant([[1,2],[3,4]])


config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
with  sess:

    ss = tf.reshape(temp,[-1]).eval()
    dnsadjksa = tf.tile(ss,[ 4]).eval()
    
    ress = tf.reshape(dnsadjksa,[6,6]).eval()
    

    one = tf.tile(tf.expand_dims(temp, 2), [1, 2, 1]).eval()
    two = tf.tile(one, [1, 1, 2]).eval()
    three = tf.reshape(tf.squeeze(two),[6,6]).eval()
    print(three)
    
    print("======77777777777777=======")
    temp = tf.random_normal([2,2,1,3]).eval()
    print(temp)
    print("======777976787656789876=======")
    temp = tf.expand_dims(temp,axis=4)
    one = tf.tile(temp, [1, 2, 1,1,1]).eval()
    two = tf.tile(one, [1, 1, 1,1,2]).eval()
    two = tf.squeeze(two).eval()
    three = tf.reshape(tf.squeeze(two),[4,4,1,3]).eval()
    print(three)


    print("======77777777777777=======")
    temp = tf.random_normal([2,2,1,3]).eval()
    print(temp)
    print("======777976787656789876=======")
    one = tf.tile(temp, [1, 2, 1,1]).eval()
    two = tf.tile(one, [2, 1, 1,1]).eval()
    three = tf.reshape(tf.squeeze(two),[4,4,1,3]).eval()
    print(three)

    
    print("======77777777777777=======")
    temp = tf.random_normal([2,2,1,3]).eval()
    print(temp)
    print("======777976787656789876=======")
    one = tf.tile(temp, [2, 2, 1,1]).eval()
    three = tf.reshape(tf.squeeze(one),[4,4,1,3]).eval()
    print(three)