
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tqdm import tqdm

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

def LSTM_layer(lstm_cell_units, number_of_layers, batch_size, dropout_rate=0.8):
    '''
    This method is used to create LSTM layer/s for PixelRNN
    
    Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
              number_of_layers - used to define how many of LSTM layers do we want in the network
              batch_size - in this method this information is used to build starting state for the network
              dropout_rate - used to define how many cells in a layer do we want to 'turn off'
              
    Output(s): cell - lstm layer
               init_state - zero vectors used as a starting state for the network
    '''
    
    
    layer = tf.contrib.rnn.BasicLSTMCell(lstm_cell_units)
    
    if dropout_rate != 0:
        layer = tf.contrib.rnn.DropoutWrapper(layer, dropout_rate)
        
    cell = tf.contrib.rnn.MultiRNNCell([layer]*number_of_layers)
    
    init_size = cell.zero_state(batch_size, tf.float32)
    
    return cell, init_size


def rnn_output(lstm_outputs, input_size, output_size):
    '''
    Output layer for the lstm netowrk
    
    Input(s): lstm_outputs - outputs from the RNN part of the network
              input_size - in this case it is RNN size (number of neuros in RNN layer)
              output_size - number of neuros for the output layer == number of classes
              
    Output(s) - logits, 
    '''
    
    
    outputs = lstm_outputs[:, -1, :]
    
    weights = tf.Variable(tf.random_uniform([input_size, output_size]), name='rnn_out_weights')
    bias = tf.Variable(tf.zeros([output_size]), name='rnn_out_bias')
    
    output_layer = tf.matmul(outputs, weights) + bias
    return output_layer


def loss_optimizer(rnn_out, targets, learning_rate):
    '''
    Function used to calculate loss and minimize it
    
    Input(s): rnn_out - logits from the fully_connected layer
              targets - targets used to train network
              learning_rate/step_size
    
    
    Output(s): optimizer - optimizer of choice
               loss - calculated loss function
    '''
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=rnn_out, labels=targets)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return optimizer, loss

class PixelRNN(object):
    
    def __init__(self, learning_rate=0.001, batch_size=100, classes=10, img_size = (28, 28), lstm_size=128,
                number_of_layers=1, dropout_rate=0.6):
        
        '''
        PixelRNN - call this class to create whole model
        
        Input(s): learning_rate - how fast are we going to move towards global minima
                  batch_size - how many samples do we feed at ones
                  classes - number of classes that we are trying to recognize
                  img_size - width and height of a single image
                  lstm_size - number of neurons in a LSTM layer
                  number_of_layers - number of RNN layers in the PixelRNN 
                  dropout_rate - % of cells in a layer that we are stopping gradients to flow through
        '''
        
        #This placeholders are just for images
        self.inputs = tf.placeholder(tf.float32, [None, img_size[0], img_size[1]], name='inputs')
        self.targets = tf.placeholder(tf.int32, [None, classes], name='targets')
        
        cell, init_state = LSTM_layer(lstm_size, number_of_layers, batch_size, dropout_rate)
        
        outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)
        
        rnn_out = rnn_output(outputs, lstm_size, classes)
        
        self.opt, self.cost = loss_optimizer(rnn_out, self.targets, learning_rate)
        
        predictions = tf.nn.softmax(rnn_out)
        
        currect_pred = tf.equal(tf.cast(tf.round(tf.argmax(predictions, 1)), tf.int32), tf.cast(tf.argmax(self.targets, 1), tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(currect_pred, tf.float32))
        
        self.predictions = tf.argmax(tf.nn.softmax(rnn_out), 1)



tf.reset_default_graph()
model = PixelRNN()
session = tf.Session()
session.run(tf.global_variables_initializer())


epochs = 20
batch_size = 100
image_vector = 28*28


for i in range(epochs):
    training_accuracy = []
    epoch_loss = []
 
    for ii in tqdm(range(mnist.train.num_examples // batch_size)):
        
        batch = mnist.train.next_batch(batch_size)
        
        images = batch[0].reshape((-1, 28, 28))
        targets = batch[1]
        
        c, _, a = session.run([model.cost, model.opt, model.accuracy], feed_dict={model.inputs: images, model.targets:targets})
        
        epoch_loss.append(c)
        training_accuracy.append(a)
        
    print("Epoch: {}/{}".format(i, epochs), " | Current loss: {}".format(np.mean(epoch_loss)),
          " | Training accuracy: {:.4f}%".format(np.mean(training_accuracy)))
    
    print('\n', 'TESTING PROCESS...')
    
    testing_accuracy = []
    for k in range(mnist.test.num_examples // batch_size):
        
        batch_test = mnist.test.next_batch(batch_size)
        
        images_test = batch[0].reshape((-1, 28, 28))
        targets_test = batch[1]
        
        ta = session.run([model.accuracy], feed_dict={model.inputs: images_test, model.targets:targets_test})
        testing_accuracy.append(ta)
        
    print("Testing accuracy is: {:.4f}%".format(np.mean(testing_accuracy)))

#  -- end code -- 