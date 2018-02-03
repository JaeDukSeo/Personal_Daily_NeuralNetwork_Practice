import tensorflow as tf
import numpy as np,os,sys,sklearn
from scipy import misc
from sklearn.metrics import confusion_matrix
# from sklearn.metrics.classification_report
import itertools
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

np.random.seed(1234)
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

aug_filp = iaa.Sequential([
    # iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    # iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

aug_crop = iaa.Sequential([
    iaa.Crop(px=(0, 20)), # crop images from each side by 0 to 16px (randomly chosen)
    # iaa.Fliplr(0.5), # horizontally flip 50% of the images
    # iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

aug_g_noise = iaa.Sequential([
    # iaa.Crop(px=(0, 20)), # crop images from each side by 0 to 16px (randomly chosen)
    # iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 2.0)) # blur images with a sigma of 0 to 3.0
])

aug_g_noise_filp = iaa.Sequential([
    # iaa.Crop(px=(0, 20)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 2.0)) # blur images with a sigma of 0 to 3.0
])

aug_crop_filp = iaa.Sequential([
    iaa.Crop(px=(0, 20)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    # iaa.GaussianBlur(sigma=(0, 2.0)) # blur images with a sigma of 0 to 3.0
])


print train_data.shape
images=train_data[5,:,:,:]
print images.shape
plt.imshow(images)
plt.show()


temp = aug_filp.augment_image(images)
plt.imshow(temp)
plt.show()
temp = aug_crop.augment_image(images)
plt.imshow(temp)
plt.show()
temp = aug_g_noise.augment_image(images)
plt.imshow(temp)
plt.show()
temp = aug_g_noise_filp.augment_image(images)
plt.imshow(temp)
plt.show()
temp = aug_crop_filp.augment_image(images)
plt.imshow(temp)
plt.show()

sys.exit()
combine_data = np.vstack((train_data,test_data))
combine_label = np.vstack((train_label,test_label))

# writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

# 1. Make the graph
graph = tf.Graph()
with graph.as_default():

    x = tf.placeholder('float',[None,100,100,3],name="x")
    y = tf.placeholder('float',[None,3],name="y")
    keep_prob = tf.placeholder('float',name="keep_prob")

    w1 = tf.Variable(tf.random_normal([5,5,3,5] ,stddev=0.01 ),name="w1")
    w2 = tf.Variable(tf.random_normal([4,4,5,7] ,stddev=0.01 ),name="w2")
    w3 = tf.Variable(tf.random_normal([3,3,7,10] ,stddev=0.01 ),name="w3")
    w4 = tf.Variable(tf.random_normal([1690,525],stddev=0.01),name="w4")
    w5 = tf.Variable(tf.random_normal([525,325],stddev=0.01),name="w5")
    w6 = tf.Variable(tf.random_normal([325,3],stddev=0.01),name="w6")

    b1 = tf.Variable(tf.random_normal([5],stddev=0.01),name="b1")
    b2 = tf.Variable(tf.random_normal([7],stddev=0.01),name="b2")
    b3 = tf.Variable(tf.random_normal([10],stddev=0.01),name="b3")

    b4 = tf.Variable(tf.random_normal([525],stddev=0.01),name="b4")
    b5 = tf.Variable(tf.random_normal([325],stddev=0.01),name="b5")
    b6 = tf.Variable(tf.random_normal([3],stddev=0.01),name="b6")

    layer_1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding="SAME",name="layer_1")
    layer_1 = tf.nn.bias_add(layer_1,b1,name='bias_1_add')
    layer_1_pool = tf.nn.max_pool(layer_1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME',name="layer_1_pool")
    layer_1_act = tf.nn.elu(layer_1_pool,name='elu_1')

    layer_2 = tf.nn.conv2d(layer_1_act,w2,strides=[1,1,1,1],padding="SAME",name="layer_2")
    layer_2 = tf.nn.bias_add(layer_2,b2,name='bias_2_add')
    layer_2_pool = tf.nn.max_pool(layer_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME',name="layer_2_pool")
    layer_2_act = tf.nn.elu(layer_2_pool,name='elu_2')

    layer_3 = tf.nn.conv2d(layer_2_act,w3,strides=[1,1,1,1],padding="SAME",name="layer_3")
    layer_3 = tf.nn.bias_add(layer_3,b3,name='bias_3_add')
    layer_3_pool = tf.nn.max_pool(layer_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME',name="layer_3_pool")
    layer_3_act = tf.nn.elu(layer_3_pool,name='elu_3')

    layer_4_data = tf.reshape(layer_3_act,[6,-1],name='layer_4_reshape_Flat')
    layer_4_dropout = tf.nn.dropout(layer_4_data,keep_prob,name="layer_4_dropout")
    layer_4 = tf.matmul(layer_4_data,w4,name='layer_4') 
    layer_4 = tf.nn.bias_add(layer_4,b4,name='bias_4_add')
    layer_4_act = tf.nn.elu(layer_4,name='elu_4')

    layer_5_dropout=  tf.nn.dropout(layer_4_act,keep_prob,name="layer_5_dropout")
    layer_5 = tf.matmul(layer_5_dropout,w5,name='layer_5')
    layer_5 = tf.nn.bias_add(layer_5,b5,name='bias_5_add')
    layer_5_act = tf.nn.elu(layer_5,name='elu_5')

    layer_6_dropout=  tf.nn.dropout(layer_5_act,keep_prob,name="layer_6_dropout")
    layer_6 = tf.matmul(layer_6_dropout,w6,name='layer_6')
    layer_6 = tf.nn.bias_add(layer_6,b6,name='bias_6_add')
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_6, labels=y),name="loss")
    optimizer = tf.train.AdamOptimizer(0.0003).minimize(loss)
    train_prediction = tf.nn.softmax(layer_6,name='softmax')

# 2. Make the Session 0002
sess = tf.Session(graph=graph)
with sess:

    writer = tf.summary.FileWriter("z_tf_Board/output", sess.graph)
    sess.run(tf.global_variables_initializer())

    for iter in range(500):

        current_x,current_label = sklearn.utils.shuffle(train_data,train_label)
        sum_loss = 0

        for i in range(0,len(current_x)):
            
            images =np.expand_dims(current_x[i],axis=0)
            current_x_data = images
            current_x_label = np.expand_dims(current_label[i],axis=0)
            temp_label = np.copy(current_x_label)
            for jj in range(0,5):
                current_x_label = np.vstack((temp_label,current_x_label))

            images_aug = aug_filp.augment_images(images)
            current_x_data = np.vstack((current_x_data,images_aug))
            images_aug = aug_crop.augment_images(images)
            current_x_data = np.vstack((current_x_data,images_aug))
            images_aug = aug_g_noise.augment_images(images)
            current_x_data = np.vstack((current_x_data,images_aug))
            images_aug = aug_g_noise_filp.augment_images(images)
            current_x_data = np.vstack((current_x_data,images_aug))
            images_aug = aug_crop_filp.augment_images(images)
            current_x_data = np.vstack((current_x_data,images_aug))

            temp_data = sess.run([layer_6,loss,optimizer,train_prediction],feed_dict={x:current_x_data,y:current_x_label,keep_prob:0.6})
            sum_loss+= temp_data[1].sum()
        print "Epoch: ",iter," Sum loss : ",sum_loss

        if iter == 80:
            learing_rate = 0.0001

        if iter == 400:
            learing_rate = 0.00005


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

    for j in range(0,len(test_data)):

        current_test =test_data[j]
        current_test_label =test_label[j]

        images =np.expand_dims(test_data[j],axis=0)
        current_x_data = images
        current_x_label = np.expand_dims(test_label[j],axis=0)

        temp_label = np.copy(current_x_label)
        for jj in range(0,5):
            current_x_label = np.vstack((temp_label,current_x_label))

        images_aug = aug_filp.augment_images(images)
        current_x_data = np.vstack((current_x_data,images_aug))
        images_aug = aug_crop.augment_images(images)
        current_x_data = np.vstack((current_x_data,images_aug))
        images_aug = aug_g_noise.augment_images(images)
        current_x_data = np.vstack((current_x_data,images_aug))
        images_aug = aug_g_noise_filp.augment_images(images)
        current_x_data = np.vstack((current_x_data,images_aug))
        images_aug = aug_crop_filp.augment_images(images)
        current_x_data = np.vstack((current_x_data,images_aug))

        temp_data = sess.run([layer_6,loss,optimizer,train_prediction],feed_dict={x:current_x_data,y:current_x_label,keep_prob:1.0})

        if j % 5 == 0:
            print "Predicted Index Max : ", np.argmax(temp_data[3],axis=1)
            print "GT Index Max : ",np.argmax(current_x_label,axis=1)
            print "Mactch : " ,( np.argmax(temp_data[3],axis=1) == np.argmax(current_x_label,axis=1))
            print "Mactch count : " , ((np.argmax(temp_data[3],axis=1) == np.argmax(current_x_label,axis=1)) * 1.0).sum()
            print '\n\n'

        total_match_gt = np.append(total_match_gt,np.argmax(current_x_label,axis=1))
        total_match_list = np.append(total_match_list,np.argmax(temp_data[3],axis=1))
        total_match += ((np.argmax(temp_data[3],axis=1) == np.argmax(current_x_label,axis=1)) * 1.0).sum()

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
    print '------- Starting Final Round Combine List -------'

    combine_data,combine_label = sklearn.utils.shuffle(combine_data,combine_label)

    total_match = 0
    total_match_list = np.array([])
    total_match_gt  = np.array([])

    for j in range(0,len(combine_data),6):

        current_test =combine_data[j:j+6]
        current_test_label =combine_label[j:j+6]

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
    print "Total Match : ",total_match,"  ",len(combine_label)
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
    







# ------ END CODE -------