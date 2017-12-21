import tensorflow as tf
import numpy as np,os,sys,sklearn
from scipy import misc
from sklearn.metrics import confusion_matrix
# from sklearn.metrics.classification_report
import itertools
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1234)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 0. Data Preprocess
normal_dir = "../1_CPS_40_Data/1_Normal_I/resized_small/"
adenoma_dir = "../1_CPS_40_Data/2_adenoma_II/resized_small/"
cancer_dir = "../1_CPS_40_Data/3_Cancer_III/resized_small/"

normal_data,adenoma_data,cancer_data =[],[],[]

image_size = 100
image_dim = image_size * image_size * 3
label_count = 3

# 1. Read all of the data first
for dirName, subdirList, fileList in os.walk(normal_dir):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM misc.imread(
            image_read = misc.imread(os.path.join(dirName,filename),mode ='RGB')
            normal_data.append(image_read)

for dirName, subdirList, fileList in os.walk(adenoma_dir):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            image_read = misc.imread(os.path.join(dirName,filename),mode ='RGB')
            adenoma_data.append(image_read)

for dirName, subdirList, fileList in os.walk(cancer_dir):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            image_read = misc.imread(os.path.join(dirName,filename),mode ='RGB')
            cancer_data.append(image_read)

# 2. Create one hot encoding
normal_hot,adenoma_hot,cancer_hot =[],[],[]
for x in range(len(normal_data)):
    normal_hot.append([1.0,0,0])
for x in range(len(adenoma_data)):
    adenoma_hot.append([0,1.0,0])
for x in range(len(cancer_data)):
    cancer_hot.append([0,0,1.0])

# 3. Create the train data train label and test data and test label
train_data,train_label = [],[]
test_data,test_label = [],[]

normal_train =  normal_data[:-10]
normal_train_label = normal_hot[:-10]
normal_test = normal_data[-10:]
normal_test_label = normal_hot[-10:]

adenoma_train =  adenoma_data[:-10]
adenoma_train_label = adenoma_hot[:-10]
adenoma_test = adenoma_data[-10:]
adenoma_test_label = adenoma_hot[-10:]

cancer_train =  cancer_data[:-10]
cancer_train_label = cancer_hot[:-10]
cancer_test = cancer_data[-10:]
cancer_test_label = cancer_hot[-10:]

train_data.extend(normal_train)
train_data.extend(adenoma_train)
train_data.extend(cancer_train)
train_label.extend(normal_train_label)
train_label.extend(adenoma_train_label)
train_label.extend(cancer_train_label)

test_data.extend(normal_test)
test_data.extend(adenoma_test)
test_data.extend(cancer_test)
test_label.extend(normal_test_label)
test_label.extend(adenoma_test_label)
test_label.extend(cancer_test_label)

train_data = np.asarray(train_data)
train_label = np.asarray(train_label)
test_data = np.asarray(test_data)
test_label = np.asarray(test_label)


# 1. Make the graph
graph = tf.Graph()
with graph.as_default():

    x = tf.placeholder('float',[None,100,100,3])
    y = tf.placeholder('float',[None,3])
    keep_prob = tf.placeholder('float')

    w1 = tf.Variable(tf.random_normal([5,5,3,5] ,stddev=0.01 ))
    w2 = tf.Variable(tf.random_normal([4,4,5,7] ,stddev=0.01 ))
    w3 = tf.Variable(tf.random_normal([3,3,7,10] ,stddev=0.01 ))
    w4 = tf.Variable(tf.random_normal([1690,525],stddev=0.01))
    w5 = tf.Variable(tf.random_normal([525,325],stddev=0.01))
    w6 = tf.Variable(tf.random_normal([325,3],stddev=0.01))

    b1 = tf.Variable(tf.random_normal([5],stddev=0.01))
    b2 = tf.Variable(tf.random_normal([7],stddev=0.01))
    b3 = tf.Variable(tf.random_normal([10],stddev=0.01))

    b4 = tf.Variable(tf.random_normal([525],stddev=0.01))
    b5 = tf.Variable(tf.random_normal([325],stddev=0.01))
    b6 = tf.Variable(tf.random_normal([3],stddev=0.01))

    layer_1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding="SAME")
    layer_1 = layer_1 + b1
    layer_1_pool = tf.nn.max_pool(layer_1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    layer_1_act = tf.nn.elu(layer_1_pool)

    layer_2 = tf.nn.conv2d(layer_1_act,w2,strides=[1,1,1,1],padding="SAME")
    layer_2 = layer_2 + b2
    layer_2_pool = tf.nn.max_pool(layer_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    layer_2_act = tf.nn.elu(layer_2_pool)

    layer_3 = tf.nn.conv2d(layer_2_act,w3,strides=[1,1,1,1],padding="SAME")
    layer_3 = layer_3 + b3
    layer_3_pool = tf.nn.max_pool(layer_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    layer_3_act = tf.nn.elu(layer_3_pool)

    layer_4_data = tf.reshape(layer_3_act,[6,-1])
    layer_4_dropout = tf.nn.dropout(layer_4_data,keep_prob)
    layer_4 = tf.matmul(layer_4_data,w4) + b4
    layer_4_act = tf.nn.elu(layer_4)

    layer_5_dropout=  tf.nn.dropout(layer_4_act,keep_prob)
    layer_5 = tf.matmul(layer_5_dropout,w5) + b5
    layer_5_act = tf.nn.elu(layer_5)

    layer_6_dropout=  tf.nn.dropout(layer_5_act,keep_prob)
    layer_6 = tf.matmul(layer_6_dropout,w6) + b6
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_6, labels=y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
    train_prediction = tf.nn.softmax(layer_6)

# 2. Make the Session
sess = tf.Session(graph=graph)
with sess:

    sess.run(tf.global_variables_initializer())


    for iter in range(100):

        current_x,current_label = sklearn.utils.shuffle(train_data,train_label)
        sum_loss = 0
        for i in range(0,len(current_x),6):
            current_x_data = current_x[i:i+6]
            current_x_label = current_label[i:i+6]
            temp_data = sess.run([layer_6,loss,optimizer,train_prediction],feed_dict={x:current_x_data,y:current_x_label,keep_prob:0.6})
            sum_loss+= temp_data[1].sum()

        print "Epoch: ",iter," Sum loss : ",sum_loss
        test_data,test_label = sklearn.utils.shuffle(test_data,test_label)
        for j in range(0,6,6):

            current_test =test_data[j:j+6]
            current_test_label =test_label[j:j+6]

            temp_data = sess.run([layer_6,loss,optimizer,train_prediction],feed_dict={x:current_test,y:current_test_label,keep_prob:1.0})
            print "Predicted Index Max : ", np.argmax(temp_data[3],axis=1)
            print "GT Index Max : ",np.argmax(current_test_label,axis=1)
            print "Mactch : " ,( np.argmax(temp_data[3],axis=1) == np.argmax(current_test_label,axis=1))
            print "Mactch count : " , ((np.argmax(temp_data[3],axis=1) == np.argmax(current_test_label,axis=1)) * 1.0).sum()
            
            print '\n\n\n'

    print '------- Starting Final Round -------'
    total_match = 0
    total_match_list = np.array([])
    total_match_gt  = np.array([])

    for j in range(0,len(test_data),6):

        current_test =test_data[j:j+6]
        current_test_label =test_label[j:j+6]

        temp_data = sess.run([layer_6,loss,optimizer,train_prediction],feed_dict={x:current_test,y:current_test_label,keep_prob:1.0})
        print "Predicted Index Max : ", np.argmax(temp_data[3],axis=1)
        print "GT Index Max : ",np.argmax(current_test_label,axis=1)
        print "Mactch : " ,( np.argmax(temp_data[3],axis=1) == np.argmax(current_test_label,axis=1))
        print "Mactch count : " , ((np.argmax(temp_data[3],axis=1) == np.argmax(current_test_label,axis=1)) * 1.0).sum()
        print '\n\n'

        total_match_gt = np.append(total_match_gt,np.argmax(current_test_label,axis=1))
        total_match_list = np.append(total_match_list,np.argmax(temp_data[3],axis=1))

        # total_match_gt.append(np.argmax(current_test_label,axis=1) )
        # total_match_list.append(np.argmax(temp_data[3],axis=1))
        total_match += ((np.argmax(temp_data[3],axis=1) == np.argmax(current_test_label,axis=1)) * 1.0).sum()


    print total_match_gt
    print total_match_list
    print "Total Match : ",total_match,"  ",len(test_label)
    cnf_matrix = confusion_matrix(total_match_gt, total_match_list)
    np.set_printoptions(precision=2)

    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Normal','Adenoma','Cancer'],title='Confusion matrix, without normalization')
    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Normal','Adenoma','Cancer'], normalize=True,title='Normalized confusion matrix')
    plt.show()
    


sys.exit()
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)





# ------ END CODE -------