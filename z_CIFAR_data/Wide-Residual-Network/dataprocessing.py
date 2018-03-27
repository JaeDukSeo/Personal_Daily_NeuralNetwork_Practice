def unpickle(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo,encoding='latin1')
    return dict

batch1 = unpickle('../cifar10/cifar-10-batches-py/data_batch_1') #Chosen as training sample
batch2 = unpickle('../cifar10/cifar-10-batches-py/data_batch_2') #Chosen as training sample
batch3 = unpickle('../cifar10/cifar-10-batches-py/data_batch_3') #Chosen as training sample
batch4 = unpickle('../cifar10/cifar-10-batches-py/data_batch_4') #Chosen as training sample
batch5 = unpickle('../cifar10/cifar-10-batches-py/data_batch_5') #Chosen as training sample
test = unpickle('../cifar10/cifar-10-batches-py/test_batch') #Chosen as test sample (same data for validation)
classes = unpickle('../cifar10/cifar-10-batches-py/batches.meta') #keyword for label = label_names

trainingSamples = len(batch1['data'])+len(batch2['data'])+len(batch3['data'])+len(batch4['data'])+len(batch5['data'])
testingSamples = len(test['data'])

print("Total Training Samples: %d"%trainingSamples)
print("Total Testing Samples: %d"%testingSamples)
print("Classes: "+str(classes['label_names']))


import numpy as np

X_train = np.zeros((trainingSamples,32,32,3))
Y_train = np.zeros((trainingSamples))
X_test = np.zeros((testingSamples,32,32,3))
Y_test = []

b1l = len(batch1['data'])
b2l = len(batch2['data'])
b3l = len(batch3['data'])
b4l = len(batch4['data'])
b5l = len(batch5['data'])
tl = len(test['data'])

# I intend to convert each batch from 10000x3072 to 10000x32x32x3 format but directly reshaping into that format
# makes each of the images strange looking because the pixel co-efficients don't get properly arranged in that way.
# Instead reshaping to 10000x3x32x32 format works fine. So that's what I did first.

b1 = batch1['data'][...].reshape((b1l,3,32,32))
b2 = batch2['data'][...].reshape((b2l,3,32,32))
b3 = batch3['data'][...].reshape((b3l,3,32,32))
b4 = batch4['data'][...].reshape((b4l,3,32,32))
b5 = batch5['data'][...].reshape((b5l,3,32,32))
t = test['data'][...].reshape((tl,3,32,32))

# Now the images are in 3x32x32 (channels first format) but I want those in 32x32x3 (channels last format)
# To do that I simply transponse the image matrices in the specific order of axes and I get the output in 
# my desired format 10000x32x32x3

b1 = np.transpose(b1,(0,2,3,1))
b2 = np.transpose(b2,(0,2,3,1))
b3 = np.transpose(b3,(0,2,3,1))
b4 = np.transpose(b4,(0,2,3,1))
b5 = np.transpose(b5,(0,2,3,1))
t = np.transpose(t,(0,2,3,1))

# (To keep things simple, I didn't make a separate validation and testing set,)

X_train = np.concatenate((b1,b2,b3,b4,b5))
Y_train = np.concatenate((batch1['labels'],batch2['labels'],batch3['labels'],batch4['labels'],batch5['labels']))


#NOT RECOMMENDED TO USE THE SAME DATA FOR TESTING AND VALIDATING!!
#I am doing it intentionally to bias this network towards test case.

X_val = t
Y_val = test['labels']
Y_val = np.array(Y_val)


X_test = t
Y_test = test['labels']
Y_test = np.array(Y_test)
        
print ("Shape of Training Samples (input and output): "+str(X_train.shape)+" "+str(Y_train.shape))
print ("Shape of Validation Samples (input and output): "+str(X_val.shape)+" "+str(Y_val.shape))
print ("Shape of Testing Samples (input and output): "+str(X_test.shape)+" "+str(Y_test.shape))



def mean_std_normalization(X,mean,calculate_mean=True):
    channel_size = X.shape[3]
    for i in range(channel_size):
        if calculate_mean == True:
            mean[i] = np.mean(X[:,:,:,i])
        variance = np.mean(np.square(X[:,:,:,i]-mean[i]))
        deviation = np.sqrt(variance)
        X[:,:,:,i] = (X[:,:,:,i]-mean[i])/deviation
    return X,mean

channel_size = X_train.shape[3]
mean = np.zeros((channel_size),dtype=np.float32)

X_train,train_mean = mean_std_normalization(X_train,mean)
X_val,_ = mean_std_normalization(X_val,train_mean,False)
X_test,_ = mean_std_normalization(X_test,train_mean,False)


#smoothing_factor = 0.1
classes_num = len(classes['label_names'])

Y_train_processed = np.zeros((len(Y_train),classes_num),np.float32)
Y_test_processed = np.zeros((len(Y_test),classes_num),np.float32)
Y_val_processed = np.zeros((len(Y_val),classes_num),np.float32)

for i in range(0,len(Y_train)):
    Y_train_processed[i][int(Y_train[i])]=1
    
#Y_train_processed = (1-smoothing_factor)*(Y_train_processed)+(smoothing_factor/classes_num)
                                                             
for i in range(0,len(Y_val)):
    Y_val_processed[i][int(Y_val[i])]=1 
    
for i in range(0,len(Y_test)):
    Y_test_processed[i][int(Y_test[i])]=1


import h5py

file = h5py.File('mean_std_cifar_10.h5','w')
file.create_dataset('X_train', data=X_train)
file.create_dataset('X_val', data=X_val)
file.create_dataset('X_test', data=X_test)
file.create_dataset('Y_train', data=Y_train_processed)
file.create_dataset('Y_val', data=Y_val_processed)
file.create_dataset('Y_test', data=Y_test_processed)
file.create_dataset('train_mean', data=train_mean)

file.close()

# -- end code --