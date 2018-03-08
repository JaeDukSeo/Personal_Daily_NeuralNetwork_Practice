import tensorflow as tf
import numpy as np,sys
from numpy import float32
import matplotlib.pyplot as plt

np.random.seed(678)
tf.set_random_seed(678)

# Activation Functions - however there was no indication in the original paper
def tf_Relu(x): return tf.nn.relu(x)
def d_tf_Relu(x): return 7 

def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf.log(x))

def tf_tanh(x): return tf.tanh(x)
def d_tf_tansh(x): return 1.0 - tf.square(tf_tanh(x))

def gaussian_noise_layer(input_layer, std=1.0):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    X = np.asarray(dict[b'data'].T).astype("uint8")
    Yraw = np.asarray(dict[b'labels'])
    Y = np.zeros((10,10000))
    for i in range(10000):
        Y[Yraw[i],i] = 1
    names = np.asarray(dict[b'filenames'])
    return X,Y,names
    # return dict

def visualize_image(X,Y,names,id):
    rgb = X[:,id]
    img = rgb.reshape(3,32,32).transpose([1, 2, 0])
    plt.imshow(img)
    plt.title(names[id])
    plt.show()

# Make each class for the networks
class prepnetwork():
    
    def __init__(self,hid_size=None):
        self.w1 = tf.Variable(tf.random_normal([3,3,1,hid_size]))
        self.w2 = tf.Variable(tf.random_normal([3,3,hid_size,hid_size]))
        self.w3 = tf.Variable(tf.random_normal([3,3,hid_size,hid_size]))
        self.w4 = tf.Variable(tf.random_normal([3,3,hid_size,hid_size]))
        self.w5 = tf.Variable(tf.random_normal([3,3,hid_size,1]))

        self.w6 = tf.Variable(tf.random_normal([3,3,1,hid_size]))
        self.w7 = tf.Variable(tf.random_normal([3,3,hid_size,hid_size]))
        self.w8 = tf.Variable(tf.random_normal([3,3,hid_size,hid_size]))
        self.w4 = tf.Variable(tf.random_normal([3,3,hid_size,hid_size]))
        self.w5 = tf.Variable(tf.random_normal([3,3,hid_size,1]))

        self.w1 = tf.Variable(tf.random_normal([3,3,1,hid_size]))
        self.w2 = tf.Variable(tf.random_normal([3,3,hid_size,hid_size]))
        self.w3 = tf.Variable(tf.random_normal([3,3,hid_size,hid_size]))
        self.w4 = tf.Variable(tf.random_normal([3,3,hid_size,hid_size]))
        self.w5 = tf.Variable(tf.random_normal([3,3,hid_size,1]))


    def getw(self): return [self.w1,self.w2,self.w3,self.w4,self.w5]

    def feedforward(self,input=None):
        layer1 = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        layer1 = tf_tanh(layer1)
        layer2 = tf.nn.conv2d(layer1,self.w2,strides=[1,1,1,1],padding='SAME')
        layer2 = tf_tanh(layer2)
        layer3 = tf.nn.conv2d(layer2,self.w3,strides=[1,1,1,1],padding='SAME')
        layer3 = tf_tanh(layer3)
        layer4 = tf.nn.conv2d(layer3,self.w4,strides=[1,1,1,1],padding='SAME')
        layer4 = tf_tanh(layer4)
        layer5 = tf.nn.conv2d(layer4,self.w5,strides=[1,1,1,1],padding='SAME')
        layer5 = tf_log(layer5)
        return layer5

class hiddingnetwork():
    
    def __init__(self,hid_size=None):
        self.w1 = tf.Variable(tf.random_normal([4,4,1,hid_size]))
        self.w2 = tf.Variable(tf.random_normal([4,4,hid_size,hid_size]))
        self.w3 = tf.Variable(tf.random_normal([4,4,hid_size,hid_size]))
        self.w4 = tf.Variable(tf.random_normal([4,4,hid_size,hid_size]))
        self.w5 = tf.Variable(tf.random_normal([4,4,hid_size,1]))

    def getw(self): return [self.w1,self.w2,self.w3,self.w4,self.w5]

    def feedforward(self,input=None):
        layer1 = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        layer1 = tf_tanh(layer1)
        layer2 = tf.nn.conv2d(layer1,self.w2,strides=[1,1,1,1],padding='SAME')
        layer2 = tf_tanh(layer2)
        layer3 = tf.nn.conv2d(layer2,self.w3,strides=[1,1,1,1],padding='SAME')
        layer3 = tf_tanh(layer3)
        layer4 = tf.nn.conv2d(layer3,self.w4,strides=[1,1,1,1],padding='SAME')
        layer4 = tf_tanh(layer4)
        layer5 = tf.nn.conv2d(layer4,self.w5,strides=[1,1,1,1],padding='SAME')
        layer5 = tf_log(layer5)
        return layer5

class revealnetwork():
    
    def __init__(self,hid_size=None):
        self.w1 = tf.Variable(tf.random_normal([5,5,1,hid_size]))
        self.w2 = tf.Variable(tf.random_normal([5,5,hid_size,hid_size]))
        self.w3 = tf.Variable(tf.random_normal([5,5,hid_size,hid_size]))
        self.w4 = tf.Variable(tf.random_normal([5,5,hid_size,hid_size]))
        self.w5 = tf.Variable(tf.random_normal([5,5,hid_size,1]))

    def getw(self): return [self.w1,self.w2,self.w3,self.w4,self.w5]

    def feedforward(self,input=None):
        layer1 = tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='SAME')
        layer1 = tf_tanh(layer1)
        layer2 = tf.nn.conv2d(layer1,self.w2,strides=[1,1,1,1],padding='SAME')
        layer2 = tf_tanh(layer2)
        layer3 = tf.nn.conv2d(layer2,self.w3,strides=[1,1,1,1],padding='SAME')
        layer3 = tf_log(layer3)
        layer4 = tf.nn.conv2d(layer3,self.w4,strides=[1,1,1,1],padding='SAME')
        layer4 = tf_tanh(layer4)
        layer5 = tf.nn.conv2d(layer4,self.w5,strides=[1,1,1,1],padding='SAME')
        layer5 = tf_log(layer5)
        return layer5

# ------- Preprocess Data --------
X,Y,names = unpickle('../../z_CIFAR_data/cifar10batchespy/data_batch_1')
X = np.reshape(X,(3,32,32,10000)).transpose([3,1,2,0])
X = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

s_images = X[:int(len(X)/2),:,:,:]
c_images = X[int(len(X)/2):,:,:,:]

print('-----------------')
print('Size for Secret Image: ',s_images.shape)
print('Size for Cover  Image: ',c_images.shape)
print('-----------------')
# s_images = X[:1000,:,:,:]
# c_images = X[1000:2000,:,:,:]

# Declare the Objects and the networks, and get the weigths for auto
hid_size = 20
print_size = 5
learing_rate = 0.01
batch_size = 10
beta_rate_for_s = 0.8

# Declare the Object and get the weights
prepnetwork = prepnetwork(hid_size)
hididingnetwork = hiddingnetwork(hid_size)
revealnetwork = revealnetwork(hid_size+5)
prepw,hidw,revw = prepnetwork.getw(),hididingnetwork.getw(),revealnetwork.getw()

#==================================================
# Make the Graph
s = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32)
c = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32)
beta = tf.placeholder(dtype=tf.float32)

# --- Prep part ---
layer_prep  = prepnetwork.feedforward(s)

# --- Cover part ---
cover_and_secret = tf.multiply(layer_prep,c)
# cover_and_secret = tf.concat([layer_prep,c],3)
c_alpha = hididingnetwork.feedforward(cover_and_secret) 

# --- Adding Noise Part Small noise, optional-----
# c_alpha_noise = gaussian_noise_layer(c_alpha)

# --- Reveal part ---
s_alpha = revealnetwork.feedforward(c_alpha) 

# --- cost -----
# c_cost = tf.square(c-c_alpha)
# s_cost = beta * tf.square(s-s_alpha)
c_cost = tf.abs(c-c_alpha)
s_cost = beta * tf.abs(s-s_alpha)
total_cost = c_cost + s_cost

# --- Auto diff ----
# auto_train_s = tf.train.AdamOptimizer(learning_rate=learing_rate2).minimize(s_cost,var_list=prepw+hidw+revw)
# auto_train_c = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(c_cost,var_list=prepw+hidw)
auto_train = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(total_cost,var_list=prepw+hidw+revw)

# --- Back Propagation for Reveal Network ----
# grad_reveal = revealnetwork.backpropagation()
#==================================================

# Start the training Session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    total_cost_current = 0
    cost_over_time = []

    for iter in range(90000):
        
        for batch_index in range(0,len(s_images),batch_size):
            current_s = np.expand_dims(float32(s_images[batch_index:batch_index+batch_size,:,:,0]),axis=3)
            current_c = np.expand_dims(float32(c_images[batch_index:batch_index+batch_size,:,:,0]),axis=3)

            # sess_results = sess.run([total_cost,auto_train_s],feed_dict={s:current_s,c:current_c,beta:beta_rate_for_s})
            # sess_results = sess.run([total_cost,auto_train_c],feed_dict={s:current_s,c:current_c,beta:beta_rate_for_s})
            
            sess_results = sess.run([total_cost,auto_train],feed_dict={s:current_s,c:current_c,beta:beta_rate_for_s})
            print('Current Iter:',iter,' Current Batch : ',batch_index,' Current cost : ',np.sum(sess_results[0]),end='\r')
            total_cost_current = total_cost_current + np.sum(sess_results[0])

        if iter % print_size == 0:
            print('\n========')
            print('Total cost: ',total_cost_current)
            print('========')
            rand_select = np.random.randint(len(current_s))
            current_s = np.expand_dims(np.expand_dims(float32(s_images[rand_select,:,:,0]),axis=3),axis=0)
            current_c = np.expand_dims(np.expand_dims(float32(c_images[rand_select,:,:,0]),axis=3),axis=0)
            sess_results = sess.run([s_alpha,c_alpha],feed_dict={s:current_s,c:current_c})

            plt.imshow(np.squeeze(current_s),cmap='gray')
            plt.savefig('images/'+str(iter) + '_s_1.png')
            plt.imshow(np.squeeze(current_c),cmap='gray')
            plt.savefig('images/'+str(iter) + '_c_2.png')
            plt.imshow(np.squeeze(sess_results[0]),cmap='gray')
            plt.savefig('images/'+str(iter) + '_s2_3.png')
            plt.imshow(np.squeeze(sess_results[1]),cmap='gray')
            plt.savefig('images/'+str(iter) + '_c2_4.png')

        cost_over_time.append(total_cost_current)
        total_cost_current = 0



# -- end code --