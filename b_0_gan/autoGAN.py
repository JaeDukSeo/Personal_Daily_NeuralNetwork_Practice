import numpy as np,sys
import tensorflow as tf,os
from sklearn.utils import shuffle
from scipy import signal
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
tf.set_random_seed(678)
np.random.seed(5678)
np.set_printoptions(precision=3,suppress=True)


def tf_log(x): return tf.sigmoid(x)
def d_tf_log(x): return tf_log(x) * (1.0 - tf_log(x))

def tf_arctan(x): return tf.atan(x)
def d_tf_arctan(x): return 1/(1+tf.square(x))

def tf_ReLU(x): return tf.nn.relu(x)
def d_tf_ReLU(x): return tf.cast(tf.greater(x, 0),dtype=tf.float32)
    
def tf_elu(x,alpha=2): return alpha*tf.nn.elu(x)
def d_tf_leu(x,alpha=2):
    one_mask  = tf.cast(tf.greater(x, 0),dtype=tf.float32)
    zero_mask = tf_elu(tf.cast(tf.less_equal(x, 0),dtype=tf.float32) * x) + alpha
    return one_mask + zero_mask

# 0. Declare Training Data and Labels
mnnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
images = np.reshape(mnnist.train.images,[mnnist.train.images.shape[0],28,28])
batch_size = 5

# 1. Declare Class
class generator():
    
    def __init__(self):

        self.w1 = tf.Variable(tf.random_normal([7,7,1,3]))
        self.w2 = tf.Variable(tf.random_normal([5,5,3,5]))
        self.w3 = tf.Variable(tf.random_normal([3,3,5,7]))

        self.w4 = tf.Variable(tf.random_normal([1792,1000]))
        self.w5 = tf.Variable(tf.random_normal([1000,1792]))

        self.w6 = tf.Variable(tf.random_normal([3,3,5,7]))
        self.w7 = tf.Variable(tf.random_normal([5,5,3,5]))
        self.w8 = tf.Variable(tf.random_normal([7,7,1,3]))
        

    def feed_forward(self,input=None):
        self.layer1 = tf_elu(tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='VALID'))
        self.layer2 = tf_elu(tf.nn.conv2d(self.layer1,self.w2,strides=[1,1,1,1],padding='VALID')  )      
        self.layer3 = tf_elu(tf.nn.conv2d(self.layer2,self.w3,strides=[1,1,1,1],padding='VALID')   )     

        self.layer4Input = tf.reshape(self.layer3,[batch_size,-1])
        self.layer4 = tf.matmul(self.layer4Input,self.w4)
        self.layer5 = tf.matmul(self.layer4,self.w5)
        self.layer6Input = tf.reshape(self.layer5,[batch_size,16,16,7])

        self.layer6 = tf_elu(tf.nn.conv2d_transpose(self.layer6Input,self.w6,strides=[1,1,1,1],padding='VALID',output_shape=[batch_size,18,18,5]) )
        self.layer7 = tf_elu(tf.nn.conv2d_transpose(self.layer6,self.w7,strides=[1,1,1,1],padding='VALID',output_shape=[batch_size,22,22,3]) )
        self.layer8 = tf_elu(tf.nn.conv2d_transpose(self.layer7,self.w8,strides=[1,1,1,1],padding='VALID',output_shape=[batch_size,28,28,1]) )
        return self.layer8

    def getw(self): return [self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8]
    
class discrimator():    
    def __init__(self):
        self.w1 = tf.Variable(tf.random_normal([7,7,1,3]))
        self.w2 = tf.Variable(tf.random_normal([5,5,3,5]))
        self.w3 = tf.Variable(tf.random_normal([3,3,5,7]))

        self.w4 = tf.Variable(tf.random_normal([1792,1000]))
        self.w5 = tf.Variable(tf.random_normal([1000,824]))
        self.w6 = tf.Variable(tf.random_normal([824,1]))
        

    def feed_forward(self,input):
        self.layer1 = tf_elu(tf.nn.conv2d(input,self.w1,strides=[1,1,1,1],padding='VALID'))
        self.layer2 = tf_elu(tf.nn.conv2d(self.layer1,self.w2,strides=[1,1,1,1],padding='VALID')  )      
        self.layer3 = tf_elu(tf.nn.conv2d(self.layer2,self.w3,strides=[1,1,1,1],padding='VALID')   )     

        self.layer4Input = tf.reshape(self.layer3,[batch_size,-1])
        self.layer4 = tf.matmul(self.layer4Input,self.w4)
        self.layer5 = tf.matmul(self.layer4,self.w5)
        self.layer6 = tf.matmul(self.layer5,self.w6)
        return self.layer6,self.layer6

    def getw(self): return [self.w1,self.w2,self.w3,self.w4,self.w5,self.w6]

# 2. Make Graph
g = generator()
d = discrimator()

theta_D = d.getw()
theta_G = g.getw()

x = tf.placeholder(shape=[None,28,28,1],dtype='float')
y = tf.placeholder(shape=[None,28,28,1],dtype='float')

G_sample = g.feed_forward(x)
D_real, D_logit_real = d.feed_forward(y)
D_fake, D_logit_fake = d.feed_forward(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer(learning_rate=0.00000001).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=0.00000001).minimize(G_loss, var_list=theta_G)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for it in range(1000000):

    X_mb, _ = mnnist.train.next_batch(batch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x: np.random.uniform(-1., 1., size=[batch_size,28, 28,1]), y:np.reshape(X_mb,[batch_size,28,28,1]) } )
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={x: np.random.uniform(-1., 1., size=[batch_size,28, 28,1]) })

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
        samples = sess.run(G_sample, feed_dict={x: np.random.uniform(-1., 1., size=[batch_size,28, 28,1]) })
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)


# # 3. Train Session
# with tf.Session() as sess: 

#     sess.run(tf.global_variables_initializer())

#     for iter in range(100):
        
#         for current_image_index in range(0,len(images),batch_size):
            
#             current_batch = np.expand_dims(images[current_image_index:current_image_index+batch_size,:,:],axis=3)
#             sess_result = sess.run()


# -- end code ...