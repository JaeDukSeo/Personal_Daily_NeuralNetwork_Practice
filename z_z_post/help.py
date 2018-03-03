import numpy as np,sys
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('seaborn')
np.set_printoptions(precision=3,suppress=True,formatter={'float': '{: 0.3f}'.format})

# -1. Tensorflow Activation functions
def tf_log(x=None):
    return tf.sigmoid(x)
def d_tf_log(x=None):
    return tf_log(x) * (1.0 - tf_log(x))

def tf_tanh(x=None):
    return tf.tanh(x)
def d_tf_tanh(x=None):
    return 1.0 - tf.square(tf.tanh(x))

def tf_arctan(x =None):
    return tf.atan(x)
def d_tf_arctan(x=None):
    return 1.0/(1.0+tf.square(x))

def tf_softmax(x=None):
    return tf.nn.softmax(x)

# 0. Allow many GPU useage - Divide the GPU useage
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth=True





names = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']



colors_bar = ['tab:green','tab:red','tab:purple','tab:grey',



        'tab:orange','tab:pink','tab:brown','aqua',



        'azure','coral','indigo','lime',



        'tan','ivory','lightblue','lavender','teal']



plt.figure()



plt.title('Frequency of Lowest Cost among total of ' + str(fundamental_checkup) + ' trials')



plt.bar(range(len(best_cost_final)), best_cost_final,color=colors_bar)



plt.xticks(range(len(best_cost_final)), names)



# plt.title(str(save_file)+'Frequency of Lowest Cost among total of ' + str(fundamental_checkup) + ' trials.png')
plt.show()


sys.exit()



# 1. Preprocess the data
mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
testing_images, testing_lables =mnist.test.images,mnist.test.labels
training_images,training_lables =mnist.train.images,mnist.train.labels


# 1.5. Global Hypereparameters
fundamental_checkup = 2
save_file = 'plots2/'
num_epoch = 801
number_of_neurons = 1024
batch_size = 1000
print_size = 100
learning_rate= 0.001
beta1,beta2 =0.9,0.999 
adam_e = 0.00000001

# 1.75 Hyperparameter for Dilated Networks and Gradient Noise
proportion_rate = 1000
decay_rate = 0.08
proportion_rate2 = 800
decay_rate2 = 0.064
proportion_rate3 = 500
decay_rate3 = 0.04
n_value = 0.001


# 2. Build a model with class
class simple_FCC():

    def __init__(self,activation=None,d_activation=None,input_dim = None,ouput_dim = None):
        self.w = tf.Variable(tf.random_normal([input_dim,ouput_dim],seed=np.random.randn()))
        self.input = None
        self.output = None
        self.activation_function,self.d_activation_function = activation,d_activation
        self.layer,self.layerA = None,None
        self.m, self.v = tf.Variable(tf.zeros([input_dim,ouput_dim],dtype=tf.float32,)),tf.Variable(tf.zeros([input_dim,ouput_dim],dtype=tf.float32,))

    def feed_forward(self,input=None):
        self.input = input
        self.layer =tf.matmul(input,self.w)
        self.layerA = self.output = self.activation_function(self.layer)
        return self.output

    def back_propagation(self,gradient= None,check=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_activation_function(self.layer)
        grad_part_w = self.input
        grad_common = tf.multiply(grad_part_1,grad_part_2)
        grad_w = tf.matmul(tf.transpose(grad_part_w),grad_common)
        grad_pass = tf.matmul(grad_common,tf.transpose(self.w))
        update_w = [tf.assign(self.w, tf.subtract(self.w,tf.multiply(learning_rate,grad_w)) )]
        return grad_pass,update_w

    def google_brain_noise(self,gradient= None,noise=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_activation_function(self.layer)
        grad_part_w = self.input
        grad_common = tf.multiply(grad_part_1,grad_part_2)
        grad_w = tf.matmul(tf.transpose(grad_part_w),grad_common)
        grad_pass = tf.matmul(grad_common,tf.transpose(self.w))
        update_w = [tf.assign(self.w, tf.subtract(self.w,tf.multiply(learning_rate,grad_w+noise)) )]
        return grad_pass,update_w

    def dilated_ADAM(self,gradient= None,check=None):
        grad_part_1 = gradient
        grad_part_2 = self.d_activation_function(self.layer)
        grad_part_w = self.input
        grad_common = tf.multiply(grad_part_1,grad_part_2)
        grad_w = tf.matmul(tf.transpose(grad_part_w),grad_common)
        grad_pass = tf.matmul(grad_common,tf.transpose(self.w))
        update_w = [tf.assign(self.m,beta1*self.m  + (1.0-beta1) *  grad_w  )]
        update_w.append(tf.assign(self.v,beta2*self.v  + (1.0-beta2) * tf.square(grad_w)  ))
        m_hat = tf.divide(self.m,1-beta1)
        v_hat = tf.divide(self.v,1-beta2)
        adam_middle = tf.divide(learning_rate,tf.add(tf.sqrt(v_hat),adam_e))
        update_w.append(tf.assign(self.w, tf.subtract(self.w,tf.multiply(adam_middle,m_hat)) ))
        return grad_pass,update_w

# Func: Case 1: Standard Gradient Descent with Noise
def case0_google_gradient_noise():



    print("\n===== Case0: Google Brain Gradient Noise with ADAM ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = 784,ouput_dim = number_of_neurons)



    layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_dil = tf.placeholder(tf.float32, shape=(None, 784))



    y_dil = tf.placeholder(tf.float32, shape=(None,10))



    iter_variable_dil = tf.placeholder(tf.float32, shape=())



    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)







    l1_dil = layer1_dil.feed_forward(x_dil)



    l2_dil = layer2_dil.feed_forward(l1_dil)



    l3_dil = layer3_dil.feed_forward(l2_dil)



    l4_dil = layer4_dil.feed_forward(l3_dil)



    l5_dil = layer5_dil.feed_forward(l4_dil)



    l5Soft_dil = tf_softmax(l5_dil)







    cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l5Soft_dil) + (1-y_dil) * tf.log(1-l5Soft_dil ) ) )



    correct_prediction = tf.equal(tf.argmax(l5Soft_dil, 1), tf.argmax(y_dil, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # ------ Calculate The Additive Noise -------



    ADDITIVE_NOISE_STD = tf.divide(n_value,tf.pow( tf.add(1.0,iter_variable_dil),0.55 ) )



    ADDITIVE_GAUSSIAN_NOISE = tf.random_normal(mean=0,stddev=ADDITIVE_NOISE_STD,shape=())



    # ------ Calculate The Additive Noise -------







    gradient5_dil,weightup_5_dil = layer5_dil.google_brain_noise(l5Soft_dil - y_dil,ADDITIVE_GAUSSIAN_NOISE)



    gradient4_dil,weightup_4_dil = layer4_dil.google_brain_noise(gradient5_dil   ,ADDITIVE_GAUSSIAN_NOISE)



    gradient3_dil,weightup_3_dil = layer3_dil.google_brain_noise(gradient4_dil   ,ADDITIVE_GAUSSIAN_NOISE)



    gradient2_dil,weightup_2_dil = layer2_dil.google_brain_noise(gradient3_dil   ,ADDITIVE_GAUSSIAN_NOISE)



    gradient1_dil,weightup_1_dil = layer1_dil.google_brain_noise(gradient2_dil   ,ADDITIVE_GAUSSIAN_NOISE)



    weight_update_dil = [weightup_5_dil,weightup_4_dil,weightup_3_dil,weightup_2_dil,weightup_1_dil]







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())



        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil,accuracy,ADDITIVE_GAUSSIAN_NOISE],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[3] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_dil:training_images,y_dil:training_lables,iter_variable_dil:iter })



            test_accuracy  = sess.run(accuracy,feed_dict={x_dil:testing_images, y_dil:testing_lables,iter_variable_dil:iter })







            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0



        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: Dilated Back Propagation Sparse Connection
def case1_dilated_sparse():



    print("\n\n===== Case1: Dilated Sparse Back Propagation ADAM Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = 784,ouput_dim = number_of_neurons)



    layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_dil = tf.placeholder(tf.float32, shape=(None, 784))



    y_dil = tf.placeholder(tf.float32, shape=(None,10))



    iter_variable_dil = tf.placeholder(tf.float32, shape=())



    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)







    l1_dil = layer1_dil.feed_forward(x_dil)



    l2_dil = layer2_dil.feed_forward(l1_dil)



    l3_dil = layer3_dil.feed_forward(l2_dil)



    l4_dil = layer4_dil.feed_forward(l3_dil)



    l5_dil = layer5_dil.feed_forward(l4_dil)



    l5Soft_dil = tf_softmax(l5_dil)







    cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l5Soft_dil) + (1-y_dil) * tf.log(1-l5Soft_dil ) ) )



    correct_prediction = tf.equal(tf.argmax(l5Soft_dil, 1), tf.argmax(y_dil, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    gradient5_dil,weightup_5_dil = layer5_dil.dilated_ADAM(l5Soft_dil - y_dil)



    gradient4_dil,weightup_4_dil = layer4_dil.dilated_ADAM(gradient5_dil)



    gradient3_dil,weightup_3_dil = layer3_dil.dilated_ADAM(gradient4_dil+decay_propotoin_rate*gradient5_dil)



    gradient2_dil,weightup_2_dil = layer2_dil.dilated_ADAM(gradient3_dil+decay_propotoin_rate*gradient4_dil)



    gradient1_dil,weightup_1_dil = layer1_dil.dilated_ADAM(gradient2_dil+decay_propotoin_rate*gradient3_dil)



    weight_update_dil = [weightup_5_dil,weightup_4_dil,weightup_3_dil,weightup_2_dil,weightup_1_dil]







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())



        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil,accuracy],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[3] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_dil:training_images,y_dil:training_lables,iter_variable_dil:iter })



            test_accuracy  = sess.run(accuracy,feed_dict={x_dil:testing_images, y_dil:testing_lables,iter_variable_dil:iter })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0



        sess.close()



            



    return cost_array,train_acc,test_acc

# Func: Dilated noise Back Propagation Sparse Connection
def case2_dilated_sparse_noise():



    print("\n\n===== Case2: Dilated Sparse Back Noise Propagation  ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = 784,ouput_dim = number_of_neurons)



    layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_dil = tf.placeholder(tf.float32, shape=(None, 784))



    y_dil = tf.placeholder(tf.float32, shape=(None,10))



    iter_variable_dil = tf.placeholder(tf.float32, shape=())



    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)







    l1_dil = layer1_dil.feed_forward(x_dil)



    l2_dil = layer2_dil.feed_forward(l1_dil)



    l3_dil = layer3_dil.feed_forward(l2_dil)



    l4_dil = layer4_dil.feed_forward(l3_dil)



    l5_dil = layer5_dil.feed_forward(l4_dil)



    l5Soft_dil = tf_softmax(l5_dil)







    cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l5Soft_dil) + (1-y_dil) * tf.log(1-l5Soft_dil ) ) )



    correct_prediction = tf.equal(tf.argmax(l5Soft_dil, 1), tf.argmax(y_dil, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # ------ Calculate The Additive Noise -------



    ADDITIVE_NOISE_STD = tf.divide(n_value,tf.pow( tf.add(1.0,iter_variable_dil),0.55 ) )



    ADDITIVE_GAUSSIAN_NOISE = tf.random_normal(mean=0,stddev=ADDITIVE_NOISE_STD,shape=())



    # ------ Calculate The Additive Noise -------







    gradient5_dil,weightup_5_dil = layer5_dil.google_brain_noise(l5Soft_dil - y_dil,ADDITIVE_GAUSSIAN_NOISE)



    gradient4_dil,weightup_4_dil = layer4_dil.google_brain_noise(gradient5_dil   ,ADDITIVE_GAUSSIAN_NOISE)



    gradient3_dil,weightup_3_dil = layer3_dil.google_brain_noise(gradient4_dil+decay_propotoin_rate*gradient5_dil   ,ADDITIVE_GAUSSIAN_NOISE)



    gradient2_dil,weightup_2_dil = layer2_dil.google_brain_noise(gradient3_dil+decay_propotoin_rate*gradient4_dil   ,ADDITIVE_GAUSSIAN_NOISE)



    gradient1_dil,weightup_1_dil = layer1_dil.google_brain_noise(gradient2_dil+decay_propotoin_rate*gradient3_dil   ,ADDITIVE_GAUSSIAN_NOISE)



    weight_update_dil = [weightup_5_dil,weightup_4_dil,weightup_3_dil,weightup_2_dil,weightup_1_dil]







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())



        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil,accuracy,ADDITIVE_GAUSSIAN_NOISE],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[3] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_dil:training_images,y_dil:training_lables,iter_variable_dil:iter })



            test_accuracy  = sess.run(accuracy,feed_dict={x_dil:testing_images, y_dil:testing_lables,iter_variable_dil:iter })







            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0



        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: Dilated Back Propagation Dense Connection by Addition
def case3_dilated_dense_add():



    print("\n\n===== Case3: Dilated Dense Add Back Propagation ADAM Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = 784,ouput_dim = number_of_neurons)



    layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_dil = tf.placeholder(tf.float32, shape=(None, 784))



    y_dil = tf.placeholder(tf.float32, shape=(None,10))



    iter_variable_dil = tf.placeholder(tf.float32, shape=())



    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)







    l1_dil = layer1_dil.feed_forward(x_dil)



    l2_dil = layer2_dil.feed_forward(l1_dil)



    l3_dil = layer3_dil.feed_forward(l2_dil)



    l4_dil = layer4_dil.feed_forward(l3_dil)



    l5_dil = layer5_dil.feed_forward(l4_dil)



    l5Soft_dil = tf_softmax(l5_dil)







    cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l5Soft_dil) + (1-y_dil) * tf.log(1-l5Soft_dil ) ) )



    correct_prediction = tf.equal(tf.argmax(l5Soft_dil, 1), tf.argmax(y_dil, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    gradient5_dil,weightup_5_dil = layer5_dil.dilated_ADAM(l5Soft_dil - y_dil)



    gradient4_dil,weightup_4_dil = layer4_dil.dilated_ADAM(gradient5_dil)



    gradient3_dil,weightup_3_dil = layer3_dil.dilated_ADAM(gradient4_dil+decay_propotoin_rate*(gradient5_dil))



    gradient2_dil,weightup_2_dil = layer2_dil.dilated_ADAM(gradient3_dil+decay_propotoin_rate*(gradient4_dil+gradient5_dil))



    gradient1_dil,weightup_1_dil = layer1_dil.dilated_ADAM(gradient2_dil+decay_propotoin_rate*(gradient3_dil+gradient4_dil+gradient5_dil))



    weight_update_dil = [weightup_5_dil,weightup_4_dil,weightup_3_dil,weightup_2_dil,weightup_1_dil]







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())



        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil,accuracy],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[3] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_dil:training_images,y_dil:training_lables,iter_variable_dil:iter })



            test_accuracy  = sess.run(accuracy,feed_dict={x_dil:testing_images, y_dil:testing_lables,iter_variable_dil:iter })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0



        sess.close()



            



    return cost_array,train_acc,test_acc

# Func: Dilated Back Propagation Dense Connection by Multiplication
def case4_dilated_dense_multiply():



    print("\n\n===== Case 4: Dilated Desne Mul Back Propagation ADAM Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = 784,ouput_dim = number_of_neurons)



    layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_dil = tf.placeholder(tf.float32, shape=(None, 784))



    y_dil = tf.placeholder(tf.float32, shape=(None,10))



    iter_variable_dil = tf.placeholder(tf.float32, shape=())



    decay_propotoin_rate = proportion_rate / (1 + decay_rate * iter_variable_dil)







    l1_dil = layer1_dil.feed_forward(x_dil)



    l2_dil = layer2_dil.feed_forward(l1_dil)



    l3_dil = layer3_dil.feed_forward(l2_dil)



    l4_dil = layer4_dil.feed_forward(l3_dil)



    l5_dil = layer5_dil.feed_forward(l4_dil)



    l5Soft_dil = tf_softmax(l5_dil)







    cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l5Soft_dil) + (1-y_dil) * tf.log(1-l5Soft_dil ) ) )



    correct_prediction = tf.equal(tf.argmax(l5Soft_dil, 1), tf.argmax(y_dil, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    gradient5_dil,weightup_5_dil = layer5_dil.dilated_ADAM(l5Soft_dil - y_dil)



    gradient4_dil,weightup_4_dil = layer4_dil.dilated_ADAM(gradient5_dil)



    gradient3_dil,weightup_3_dil = layer3_dil.dilated_ADAM(gradient4_dil+decay_propotoin_rate*gradient5_dil)



    gradient2_dil,weightup_2_dil = layer2_dil.dilated_ADAM(gradient3_dil+decay_propotoin_rate*gradient4_dil*gradient5_dil)



    gradient1_dil,weightup_1_dil = layer1_dil.dilated_ADAM(gradient2_dil+decay_propotoin_rate*gradient3_dil*gradient4_dil*gradient5_dil)



    weight_update_dil = [weightup_5_dil,weightup_4_dil,weightup_3_dil,weightup_2_dil,weightup_1_dil]







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())



        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil,accuracy],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[3] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_dil:training_images,y_dil:training_lables,iter_variable_dil:iter })



            test_accuracy  = sess.run(accuracy,feed_dict={x_dil:testing_images, y_dil:testing_lables,iter_variable_dil:iter })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0



        sess.close()



            



    return cost_array,train_acc,test_acc

# Func: Dilated Noise Back Propagation Dense Connection by Addition
def case5_dilated_dense_add2():



    print("\n\n===== Case 5: Dilated Dense Add 2 Back Propagation 2 ADAM Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = 784,ouput_dim = number_of_neurons)



    layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_dil = tf.placeholder(tf.float32, shape=(None, 784))



    y_dil = tf.placeholder(tf.float32, shape=(None,10))



    iter_variable_dil = tf.placeholder(tf.float32, shape=())



    decay_propotoin_rate = proportion_rate2 / (1 + decay_rate2 * iter_variable_dil)







    l1_dil = layer1_dil.feed_forward(x_dil)



    l2_dil = layer2_dil.feed_forward(l1_dil)



    l3_dil = layer3_dil.feed_forward(l2_dil)



    l4_dil = layer4_dil.feed_forward(l3_dil)



    l5_dil = layer5_dil.feed_forward(l4_dil)



    l5Soft_dil = tf_softmax(l5_dil)







    cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l5Soft_dil) + (1-y_dil) * tf.log(1-l5Soft_dil ) ) )



    correct_prediction = tf.equal(tf.argmax(l5Soft_dil, 1), tf.argmax(y_dil, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    gradient5_dil,weightup_5_dil = layer5_dil.dilated_ADAM(l5Soft_dil - y_dil)



    gradient4_dil,weightup_4_dil = layer4_dil.dilated_ADAM(gradient5_dil)



    gradient3_dil,weightup_3_dil = layer3_dil.dilated_ADAM(gradient4_dil+decay_propotoin_rate*(gradient5_dil))



    gradient2_dil,weightup_2_dil = layer2_dil.dilated_ADAM(gradient3_dil+decay_propotoin_rate*(gradient4_dil+gradient5_dil))



    gradient1_dil,weightup_1_dil = layer1_dil.dilated_ADAM(gradient2_dil+decay_propotoin_rate*(gradient3_dil+gradient4_dil+gradient5_dil))



    weight_update_dil = [weightup_5_dil,weightup_4_dil,weightup_3_dil,weightup_2_dil,weightup_1_dil]







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())



        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil,accuracy],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[3] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_dil:training_images,y_dil:training_lables,iter_variable_dil:iter })



            test_accuracy  = sess.run(accuracy,feed_dict={x_dil:testing_images, y_dil:testing_lables,iter_variable_dil:iter })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0



        sess.close()



            



    return cost_array,train_acc,test_acc

# Func: Dilated Noise Back Propagation Dense Connection by Multiplication
def case6_dilated_dense_add3():



    print("\n\n===== Case 6: Dilated Dense Add 3 Back Propagation ADAM Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = 784,ouput_dim = number_of_neurons)



    layer2_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_dil = simple_FCC(tf_arctan,d_tf_arctan,    input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_dil = tf.placeholder(tf.float32, shape=(None, 784))



    y_dil = tf.placeholder(tf.float32, shape=(None,10))



    iter_variable_dil = tf.placeholder(tf.float32, shape=())



    decay_propotoin_rate = proportion_rate3 / (1 + decay_rate3 * iter_variable_dil)







    l1_dil = layer1_dil.feed_forward(x_dil)



    l2_dil = layer2_dil.feed_forward(l1_dil)



    l3_dil = layer3_dil.feed_forward(l2_dil)



    l4_dil = layer4_dil.feed_forward(l3_dil)



    l5_dil = layer5_dil.feed_forward(l4_dil)



    l5Soft_dil = tf_softmax(l5_dil)







    cost_dil = tf.reduce_sum( -1 * ( y_dil*tf.log(l5Soft_dil) + (1-y_dil) * tf.log(1-l5Soft_dil ) ) )



    correct_prediction = tf.equal(tf.argmax(l5Soft_dil, 1), tf.argmax(y_dil, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    gradient5_dil,weightup_5_dil = layer5_dil.dilated_ADAM(l5Soft_dil - y_dil)



    gradient4_dil,weightup_4_dil = layer4_dil.dilated_ADAM(gradient5_dil)



    gradient3_dil,weightup_3_dil = layer3_dil.dilated_ADAM(gradient4_dil+decay_propotoin_rate*gradient5_dil)



    gradient2_dil,weightup_2_dil = layer2_dil.dilated_ADAM(gradient3_dil+decay_propotoin_rate*(gradient4_dil+gradient5_dil))



    gradient1_dil,weightup_1_dil = layer1_dil.dilated_ADAM(gradient2_dil+decay_propotoin_rate*(gradient3_dil*gradient4_dil*gradient5_dil))



    weight_update_dil = [weightup_5_dil,weightup_4_dil,weightup_3_dil,weightup_2_dil,weightup_1_dil]







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())



        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_dil,gradient1_dil,weight_update_dil,accuracy],feed_dict={x_dil:current_batch,y_dil:current_label,iter_variable_dil:iter })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[3] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_dil:training_images,y_dil:training_lables,iter_variable_dil:iter })



            test_accuracy  = sess.run(accuracy,feed_dict={x_dil:testing_images, y_dil:testing_lables,iter_variable_dil:iter })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0



        sess.close()



            



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#GradientDescentOptimizer
def case7_auto_GradientDescentOptimizer():



    print("\n\n===== Case 7: Auto GradientDescentOptimizer Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    auto_dif = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#AdadeltaOptimizer
def case8_auto_AdadeltaOptimizer():



    print("\n\n===== Case 8: Auto AdadeltaOptimizer Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    auto_dif = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#AdagradOptimizer
def case9_auto_AdagradOptimizer():



    print("\n\n===== Case 9: Auto AdagradOptimizer Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    auto_dif = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#AdagradDAOptimizer
def case10_auto_AdagradDAOptimizer():



    print("\n\n===== Case 10: Auto AdagradDAOptimizer Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    global_step = tf.Variable(0, name='global_step',trainable=False,dtype=tf.int64)



    auto_dif = tf.train.AdagradDAOptimizer(learning_rate=learning_rate,global_step=global_step).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#MomentumOptimizer
def case11_auto_MomentumOptimizer():



    print("\n\n===== Case 11: Auto MomentumOptimizer Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    auto_dif = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=learning_rate).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



        



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#AdamOptimizer
def case12_auto_AdamOptimizer():



    print("\n\n===== Case 12: Auto ADAM Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    auto_dif = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#FtrlOptimizer
def case13_auto_FtrlOptimizer():



    print("\n\n===== Case 13: Auto FtrlOptimizer Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    auto_dif = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#ProximalGradientDescentOptimizer
def case14_auto_ProximalGradientDescentOptimizer():



    print("\n\n===== Case 14: Auto ProximalGradientDescentOptimize Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    auto_dif = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#ProximalAdagradOptimizer
def case15_auto_ProximalAdagradOptimizer():



    print("\n\n===== Case number_of_neurons: Auto ProximalAdagradOptimizer Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    auto_dif = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc

# Func: https://www.tensorflow.org/versions/r0.12/api_docs/python/train/optimizers#RMSPropOptimizer
def case16_auto_RMSPropOptimizer():



    print("\n\n===== Case 16: Auto RMSPropOptimizer Optimizer ====")



    tf.reset_default_graph()



    cost_array = []



    train_acc,test_acc = [],[]



    total_cost = 0 







    # 3. Delcare the Model



    layer1_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = 784, ouput_dim = number_of_neurons)



    layer2_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer3_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer4_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = number_of_neurons)



    layer5_auto = simple_FCC(tf_arctan,d_tf_arctan,      input_dim = number_of_neurons,ouput_dim = 10)







    # 4. Build a Graph



    x_auto = tf.placeholder(tf.float32, shape=(None, 784))



    y_auto = tf.placeholder(tf.float32, shape=(None,10))







    l1_auto = layer1_auto.feed_forward(x_auto)



    l2_auto = layer2_auto.feed_forward(l1_auto)



    l3_auto = layer3_auto.feed_forward(l2_auto)



    l4_auto = layer4_auto.feed_forward(l3_auto)



    l5_auto = layer5_auto.feed_forward(l4_auto)



    l5Soft_auto = tf_softmax(l5_auto)







    cost_auto = tf.reduce_sum( -1 * ( y_auto*tf.log(l5Soft_auto) + (1-y_auto) * tf.log(1-l5Soft_auto ) ) )



    auto_dif = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost_auto)



    correct_prediction = tf.equal(tf.argmax(l5Soft_auto, 1), tf.argmax(y_auto, 1))



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







    # 4. Run the session - Manual



    with tf.Session(config=config) as sess:







        sess.run(tf.global_variables_initializer())







        for iter in range(num_epoch) :



            



            # a. Train for all images in the training Set



            for current_image_batch in range(0,len(training_images),batch_size):



                



                current_batch = training_images[current_image_batch:current_image_batch+batch_size,:]



                current_label = training_lables[current_image_batch:current_image_batch+batch_size,:]







                sess_result = sess.run([cost_auto,auto_dif,accuracy],feed_dict={x_auto:current_batch,y_auto:current_label })



                total_cost = total_cost +sess_result[0]



                print("Real Time Cost Update Iter: ",iter, "  Cost : ",sess_result[0]," Accuracy: ",sess_result[2] ,end='\r' )







            # b. Calculated the Accuracy on both training images and test images



            train_accuracy = sess.run(accuracy,feed_dict={x_auto:training_images,y_auto:training_lables })



            test_accuracy  = sess.run(accuracy,feed_dict={x_auto:testing_images, y_auto:testing_lables })



            



            # c. Store them in list for viz



            cost_array.append(total_cost/len(training_images))



            train_acc.append(train_accuracy)



            test_acc.append(test_accuracy)







            if iter % print_size == 0 or iter == num_epoch - 1 :



                print("Current Iter: {0:4d}".format(iter), " Current total cost: {0:.4f}".format(total_cost),' Accuracy on Train: {0:.4f}'.format(train_accuracy), " Accuracy on Test: {0:.4f}".format(test_accuracy))







            # e. Reset the cost rate



            total_cost = 0







        sess.close()



    



    return cost_array,train_acc,test_acc







# Func: Start running the code



if __name__ == "__main__":







    best_cost_final  = [0] * 17



    best_train_final = [0] * 17



    best_test_final  = [0] * 17







    for law in range(fundamental_checkup):







        print("====================Starting " + str(law)+ " run of the code ====================")



        # Func: Train all of the cases to see what is best



        case0 = case0_google_gradient_noise()



        case1 = case1_dilated_sparse()



        case2 = case2_dilated_sparse_noise()



        case3 = case3_dilated_dense_add()



        case4 = case4_dilated_dense_multiply()



        case5 = case5_dilated_dense_add2()



        case6 = case6_dilated_dense_add3()



        case7 = case7_auto_GradientDescentOptimizer()



        case8 = case8_auto_AdadeltaOptimizer()



        case9 = case9_auto_AdagradOptimizer()



        case10 = case10_auto_AdagradDAOptimizer()



        case11 = case11_auto_MomentumOptimizer()



        case12 = case12_auto_AdamOptimizer()



        case13 = case13_auto_FtrlOptimizer()



        case14 = case14_auto_ProximalGradientDescentOptimizer()



        case15 = case15_auto_ProximalAdagradOptimizer()



        case16 = case16_auto_RMSPropOptimizer()



        



        # ============= Vizualize ======================



        plt.figure()



        plt.title(str(law)+' : Run of the code, Cost Over Time Graph')



        plt.plot(range(len(case1[0])),case0[0],color='tab:green', label='case 0 : Google brain Noise')



        plt.plot(range(len(case1[0])),case1[0],color='tab:red', label='case 1 : Dilated Sparse')



        plt.plot(range(len(case1[0])),case2[0],color='tab:purple', label='case 2 : Dilated Sparse Noise')



        plt.plot(range(len(case1[0])),case3[0],color='tab:gray', label='case 3 : Dilated Dense Add')



        plt.plot(range(len(case1[0])),case4[0],color='tab:orange', label='case 4 : Dilated Dense Mul')



        plt.plot(range(len(case1[0])),case5[0],color='tab:pink', label='case 5 : Dilated Dense Add 2')



        plt.plot(range(len(case1[0])),case6[0],color='tab:brown', label='case 5 : Dilated Dense Add 3')



        plt.plot(range(len(case1[0])),case7[0],color='aqua',label='case 7 : Auto GradientDescentOptimizer')



        plt.plot(range(len(case1[0])),case8[0],color='azure',label='case 8 : Auto Ada Dela')



        plt.plot(range(len(case1[0])),case9[0],color='coral',label='case 9 : Auto Ada Grad')



        plt.plot(range(len(case1[0])),case10[0],color='indigo',label='case 10 : Auto AdagradDAOptimizer')



        plt.plot(range(len(case1[0])),case11[0],color='lime',label='case 11 : Auto MomentumOptimizer')



        plt.plot(range(len(case1[0])),case12[0],color='tan',label='case 12 : Auto ADAM')



        plt.plot(range(len(case1[0])),case13[0],color='ivory',label='case 13 : Auto Ftrl')



        plt.plot(range(len(case1[0])),case14[0],color='lightblue',label='case 14 : Auto Prox Gradient')



        plt.plot(range(len(case1[0])),case15[0],color='lavender',label='case number_of_neurons : Auto Prox Ada grad')



        plt.plot(range(len(case1[0])),case16[0],color='teal',label='case 16 : Auto Prox RMSP')



        plt.legend(loc=2)



        plt.show()



        plt.figure()



        plt.title(str(law)+' : Run of the code, Accuracy on Train Image')



        plt.plot(range(len(case1[1])),case0[1],color='tab:green', label='case 0 : Google brain Noise')



        plt.plot(range(len(case1[1])),case1[1],color='tab:red', label='case 1 : Dilated Sparse')



        plt.plot(range(len(case1[1])),case2[1],color='tab:purple', label='case 2 : Dilated Sparse Noise')



        plt.plot(range(len(case1[1])),case3[1],color='tab:gray', label='case 3 : Dilated Dense Add')



        plt.plot(range(len(case1[1])),case4[1],color='tab:orange', label='case 4 : Dilated Dense Mul')



        plt.plot(range(len(case1[1])),case5[1],color='tab:pink', label='case 5 : Dilated Dense Add 2')



        plt.plot(range(len(case1[1])),case6[1],color='tab:brown', label='case 5 : Dilated Dense Add 3')



        plt.plot(range(len(case1[1])),case7[1],color='aqua',label='case 7 : Auto GradientDescentOptimizer')



        plt.plot(range(len(case1[1])),case8[1],color='azure',label='case 8 : Auto Ada Dela')



        plt.plot(range(len(case1[1])),case9[1],color='coral',label='case 9 : Auto Ada Grad')



        plt.plot(range(len(case1[1])),case10[1],color='indigo',label='case 10 : Auto AdagradDAOptimizer')



        plt.plot(range(len(case1[1])),case11[1],color='lime',label='case 11 : Auto MomentumOptimizer')



        plt.plot(range(len(case1[1])),case12[1],color='tan',label='case 12 : Auto ADAM')



        plt.plot(range(len(case1[1])),case13[1],color='ivory',label='case 13 : Auto Ftrl')



        plt.plot(range(len(case1[1])),case14[1],color='lightblue',label='case 14 : Auto Prox Gradient')



        plt.plot(range(len(case1[1])),case15[1],color='lavender',label='case number_of_neurons : Auto Prox Ada grad')



        plt.plot(range(len(case1[1])),case16[1],color='teal',label='case 16 : Auto Prox RMSP')



        plt.legend(loc=2)




        plt.show()






        plt.figure()



        plt.title(str(law)+ ' : Run of the code, Accuracy on Test Image')



        plt.plot(range(len(case1[2])),case0[2],color='tab:green', label='case 0 : Google brain Noise')



        plt.plot(range(len(case1[2])),case1[2],color='tab:red', label='case 1 : Dilated Sparse')



        plt.plot(range(len(case1[2])),case2[2],color='tab:purple', label='case 2 : Dilated Sparse Noise')



        plt.plot(range(len(case1[2])),case3[2],color='tab:gray', label='case 3 : Dilated Dense Add')



        plt.plot(range(len(case1[2])),case4[2],color='tab:orange', label='case 4 : Dilated Dense Mul')



        plt.plot(range(len(case1[2])),case5[2],color='tab:pink', label='case 5 : Dilated Dense Add 2')



        plt.plot(range(len(case1[2])),case6[2],color='tab:brown', label='case 6 : Dilated Dense Add 3')



        plt.plot(range(len(case1[2])),case7[2],color='aqua',label='case 7 : Auto GradientDescentOptimizer')



        plt.plot(range(len(case1[2])),case8[2],color='azure',label='case 8 : Auto Ada Dela')



        plt.plot(range(len(case1[2])),case9[2],color='coral',label='case 9 : Auto Ada Grad')



        plt.plot(range(len(case1[2])),case10[2],color='indigo',label='case 10 : Auto AdagradDAOptimizer')



        plt.plot(range(len(case1[2])),case11[2],color='lime',label='case 11 : Auto MomentumOptimizer')



        plt.plot(range(len(case1[2])),case12[2],color='tan',label='case 12 : Auto ADAM')



        plt.plot(range(len(case1[2])),case13[2],color='ivory',label='case 13 : Auto Ftrl')



        plt.plot(range(len(case1[2])),case14[2],color='lightblue',label='case 14 : Auto Prox Gradient')



        plt.plot(range(len(case1[2])),case15[2],color='lavender',label='case 15 : Auto Prox Ada grad')



        plt.plot(range(len(case1[2])),case16[2],color='teal',label='case 16 : Auto Prox RMSP')



        plt.legend(loc=2)



        plt.show()



        # ============= Vizualize ======================







        # ============ Get the Final Cost Final Acuuracy =========



        cost_over_time_all = [case0[0][-1],case1[0][-1],case2[0][-1],case3[0][-1],case4[0][-1],case5[0][-1],case6[0][-1],case7[0][-1],case8[0][-1],



        case9[0][-1],case10[0][-1],case11[0][-1],case12[0][-1],case13[0][-1],case14[0][-1],case15[0][-1],case16[0][-1]]



        



        train_accuracy_all = [case0[1][-1],case1[1][-1],case2[1][-1],case3[1][-1],case4[1][-1],case5[1][-1],case6[1][-1],case7[1][-1],



        case8[1][-1],case9[1][-1],case10[1][-1],case11[1][-1],case12[1][-1],case13[1][-1],case14[1][-1],case15[1][-1],case16[1][-1]]



        



        testt_accuracy_all = [case0[2][-1],case1[2][-1],case2[2][-1],case3[2][-1],case4[2][-1],case5[2][-1],case6[2][-1],case7[2][-1],



        case8[2][-1],case9[2][-1],case10[2][-1],case11[2][-1],case12[2][-1],case13[2][-1],case14[2][-1],case15[2][-1],case16[2][-1]]







        print('-----------',len(testt_accuracy_all),'-----------')







        best_cost  = cost_over_time_all.index(min(cost_over_time_all))



        best_train = train_accuracy_all.index(max(train_accuracy_all))



        best_test  = testt_accuracy_all.index(max(testt_accuracy_all))







        best_cost_final[best_cost] += 1



        best_train_final[best_train] += 1



        best_test_final[best_test] += 1







        plt.figure()



        plt.text(0.0, 0.8, str(law)+' Run Lowest Cost (divided by # of train imaegs) Case: '+ str(best_cost) + " of value" + str(cost_over_time_all[best_cost]), dict(size=30))



        plt.text(0.0, 0.5, str(law)+' Run highest Train Accuracy Case: '+ str(best_train)+ " of value" + str(train_accuracy_all[best_train]), dict(size=30))



        plt.text(0.0, 0.2, str(law)+' Run highest Test Accuracy Case: '+ str(best_test)+ " of value" + str(testt_accuracy_all[best_test]), dict(size=30))



        plt.title(str(save_file)+str(law)+ '_Run of the code, summary.png')
        plt.show()


        # ============ Get the Final Cost Final Acuuracy =========



        print("\n==================== Ending " + str(law)+ " run of the code ====================")







    # --- Final Bar graph of what was the best ------



    names = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']



    colors_bar = ['tab:green','tab:red','tab:purple','tab:grey',



                'tab:orange','tab:pink','tab:brown','aqua',



                'azure','coral','indigo','lime',



                'tan','ivory','lightblue','lavender','teal']



    plt.figure()



    plt.title('Frequency of Lowest Cost among total of ' + str(fundamental_checkup) + ' trials')



    plt.bar(range(len(best_cost_final)), best_cost_final,color=colors_bar)



    plt.xticks(range(len(best_cost_final)), names)



    # plt.title(str(save_file)+'Frequency of Lowest Cost among total of ' + str(fundamental_checkup) + ' trials.png')
    plt.show()






    plt.figure()



    plt.title('Frequency of Highest Accuracy on Train Images among total of ' + str(fundamental_checkup) + ' trials')



    plt.bar(range(len(best_train_final)), best_train_final,color=colors_bar)



    plt.xticks(range(len(best_train_final)), names)



    # plt.savefig(str(save_file)+'Frequency of  Highest Accuracy on Train Images among total of ' + str(fundamental_checkup) + ' trials.png', bbox_inches='tight')

    plt.show()






    plt.figure()



    plt.title('Frequency of Highest Accuracy on Test Images among total of ' + str(fundamental_checkup) + ' trials')



    plt.bar(range(len(best_test_final)), best_test_final,color=colors_bar)



    plt.xticks(range(len(best_test_final)), names)



    # plt.savefig(str(save_file)+'Frequency of Highest Accuracy on Test Images among total of ' + str(fundamental_checkup) + ' trials.png', bbox_inches='tight')

    plt.show()






# -- end code --