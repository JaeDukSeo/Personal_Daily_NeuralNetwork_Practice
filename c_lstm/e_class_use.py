from PIL import Image
import numpy as np,os,sys
from matplotlib import pyplot as plt
from scipy.signal import convolve2d,convolve
from sklearn.utils import shuffle

np.random.seed(563378)
np.set_printoptions(precision=3,suppress=True)


# -1. Activation Functions
def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2
def ReLu(x):
    mask = (x>0) * 1.0
    return mask *x
def d_ReLu(x):
    mask = (x>0) * 1.0
    return mask 
def log(x):
    return 1 / (1 + np.exp(-1 * x))
def d_log(x):
    return log(x) * ( 1 - log(x))
def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1 + x ** 2)






# 0. Read the data
Pathgif = "./data/dog_preprocessed/"
dog_gif = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Pathgif):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            dog_gif.append(os.path.join(dirName,filename))

Pathgif = "./data/baby_preprocessed/"
baby_gif = []  # create an empty list
for dirName, subdirList, fileList in os.walk(Pathgif):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            baby_gif.append(os.path.join(dirName,filename))

dog_array = np.zeros((24,100,100,3))
store_index = 0
for element in dog_gif:
    img = Image.open(element) 
    for iter in range(img.n_frames):
        img.seek(iter)
        new_frame = np.array(img.convert('RGB'))
        dog_array[store_index,:,:,:] = new_frame
        store_index = store_index + 1

baby_array = np.zeros((24,100,100,3))
store_index = 0
for element in baby_gif:
    img = Image.open(element) 
    for iter in range(img.n_frames):
        img.seek(iter)
        new_frame = np.array(img.convert('RGB'))
        baby_array[store_index,:,:,:] = new_frame
        store_index = store_index + 1

# 1.5 Create label and slipt the training data as well as label data
dog_label =  np.ones((24,1))
baby_label = np.zeros((24,1))
train_num =  18
train_data = np.vstack((dog_array[:train_num,:,:,:],baby_array[:train_num,:,:,:]))
train_label= np.vstack((dog_label[:train_num,:],baby_label[:train_num,:]))
test_data  = np.vstack((dog_array[train_num:,:,:,:],baby_array[train_num:,:,:,:]))
test_label = np.vstack((dog_label[train_num:,:],baby_label[train_num:,:]))

# 2. Declare the weights
num_epoch = 30
global wfx,wfrec,wix,wirec,wcx,wcrec,wox,worec
wfx =   np.random.randn(3,3,1) * 0.1
wfrec = np.random.randn(3,3,1) * 0.1

wix =   np.random.randn(3,3,1) * 0.1
wirec = np.random.randn(3,3,1) * 0.1

wcx   = np.random.randn(3,3,1) * 0.1
wcrec = np.random.randn(3,3,1) * 0.1

wox =   np.random.randn(3,3,1) * 0.1
worec = np.random.randn(3,3,1) * 0.1

w_final_1 = np.random.randn(30000,1000) * 0.001
w_final_2 = np.random.randn(1000,1) * 0.001
# hidden_state = np.random.randn(4,100,100,3) * 0.001
# cell_state = np.random.randn(4,100,100,3)   * 0.01
hidden_state = np.zeros((4,100,100,3))
cell_state   = np.zeros((4,100,100,3))
learning_rate,learning_rate2 = 0.000003,0.000003
lr3 = 0.003
total_cost = 0
cost_array = []


# 1.5 Make Class of LSTM Decoupled
class Decoupled_LSTM_Layer:
    
    def __init__(self,layer_index):
        self.wox_syth    = np.random.randn(3,3,1) * 0.1
        self.worec_syth  = np.random.randn(3,3,1) * 0.1
        
        self.wcx_syth  = np.random.randn(3,3,1) * 0.1
        self.wcrec_syth  = np.random.randn(3,3,1) * 0.1
        
        self.wix_syth  = np.random.randn(3,3,1) * 0.1
        self.wirec_syth  = np.random.randn(3,3,1) * 0.1

        self.wfx_syth  = np.random.randn(3,3,1) * 0.1
        self.wfrec_syth  = np.random.randn(3,3,1) * 0.1
        
        self.layer_index = layer_index
        self.layer_output_save = None

        self.layer_out_syth_part_x_save      = None
        self.layer_out_syth_part_rec_save    = None

        self.layer_cell_syth_part_x_save     = None
        self.layer_cell_syth_part_rec_save   = None
        
        self.layer_input_syth_part_x_save    = None
        self.layer_input_syth_part_rec_save  = None
        
        self.layer_forget_syth_part_x_save   = None
        self.layer_forget_syth_part_rec_save = None
        
    def feed_forward_synthetic_update(self,x,hidden):
        
        global wfx,wfrec,wix,wirec,wcx,wcrec,wox,worec
        layer_forget =  convolve(x,wfx,'same') + convolve(hidden,worec,'same')
        layer_forgetA = log(layer_forget)

        layer_input =   convolve(x,wix,'same') + convolve(hidden,wirec,'same')
        layer_inputA =  log(layer_input)

        layer_cell =    convolve(x,wcx,'same') + convolve(hidden,wcrec,'same')
        layer_cellA =   tanh(layer_cell)

        cell_state[self.layer_index,:,:,:] = cell_state[self.layer_index-1,:,:,:] * layer_forgetA + layer_inputA * layer_cellA

        layer_out =     convolve(x,wox,'same') + convolve(hidden,worec,'same')
        layer_outA =    log(layer_out)
        
        layer_output = layer_outA * tanh(cell_state[self.layer_index,:,:,:])
        self.layer_output_save = layer_output

        layer_out_syth_part_x =   convolve(layer_output,np.rot90(self.wox_syth,2),'same') 
        self.layer_out_syth_part_x_save      = layer_out_syth_part_x
        layer_out_syth_part_rec = convolve(layer_output,np.rot90(self.worec_syth,2),'same') 
        self.layer_out_syth_part_rec_save    = layer_out_syth_part_rec
        layer_out_syth_part_2 = tanh(cell_state[self.layer_index,:,:,:]) * d_log(layer_out) 
        layer_out_syth_x =   np.rot90(convolve(np.pad(x,((1,1),(1,1),(0,0)),'constant'),     np.rot90(layer_out_syth_part_x * layer_out_syth_part_2,2),'valid'),2)
        layer_out_syth_rec = np.rot90(convolve(np.pad(hidden,((1,1),(1,1),(0,0)),'constant'),np.rot90(layer_out_syth_part_rec * layer_out_syth_part_2,2),'valid'),2)

        layer_fic_common = layer_outA * d_tanh(cell_state[self.layer_index,:,:,:])

        layer_cell_syth_part_x   = convolve(layer_output,np.rot90(self.wcx_syth,2),'same') 
        self.layer_cell_syth_part_x_save     = layer_cell_syth_part_x
        layer_cell_syth_part_rec = convolve(layer_output,np.rot90(self.wcrec_syth,2),'same') 
        self.layer_cell_syth_part_rec_save   = layer_cell_syth_part_rec
        layer_cell_syth_part_2 = layer_fic_common * layer_inputA * d_tanh(layer_cell) 
        layer_cell_syth_x =   np.rot90(convolve(np.pad(x,((1,1),(1,1),(0,0)),'constant'),np.rot90(layer_cell_syth_part_x * layer_cell_syth_part_2,2),'valid'),2)
        layer_cell_syth_rec = np.rot90(convolve(np.pad(hidden,((1,1),(1,1),(0,0)),'constant'),np.rot90(layer_cell_syth_part_rec * layer_cell_syth_part_2,2),'valid'),2)

        layer_input_syth_part_x   = convolve(layer_output,np.rot90(self.wix_syth,2),'same') 
        self.layer_input_syth_part_x_save    = layer_input_syth_part_x
        layer_input_syth_part_rec = convolve(layer_output,np.rot90(self.wirec_syth,2),'same') 
        self.layer_input_syth_part_rec_save  = layer_input_syth_part_rec
        layer_input_syth_part_2 = layer_fic_common * layer_cellA * d_log(layer_input) 
        layer_input_syth_x =   np.rot90(convolve(np.pad(x,((1,1),(1,1),(0,0)),'constant'),np.rot90(layer_input_syth_part_x * layer_input_syth_part_2,2),'valid'),2)
        layer_input_syth_rec = np.rot90(convolve(np.pad(hidden,((1,1),(1,1),(0,0)),'constant'),np.rot90(layer_input_syth_part_rec * layer_input_syth_part_2,2),'valid'),2)

        layer_forget_syth_part_x   = convolve(layer_output,np.rot90(self.wfx_syth,2),'same') 
        self.layer_forget_syth_part_x_save   = layer_forget_syth_part_x
        layer_forget_syth_part_rec = convolve(layer_output,np.rot90(self.wfrec_syth,2),'same') 
        self.layer_forget_syth_part_rec_save = layer_forget_syth_part_rec
        layer_forget_syth_part_2 = layer_fic_common * cell_state[self.layer_index-1,:,:,:] * d_log(layer_forget) 
        layer_forget_syth_x =   np.rot90(convolve(np.pad(x,((1,1),(1,1),(0,0)),'constant'),np.rot90(layer_forget_syth_part_x * layer_forget_syth_part_2,2),'valid'),2)
        layer_forget_syth_rec = np.rot90(convolve(np.pad(hidden,((1,1),(1,1),(0,0)),'constant'),np.rot90(layer_forget_syth_part_rec * layer_forget_syth_part_2,2),'valid'),2)

        wfx   = wfx - learning_rate * layer_forget_syth_x
        wfrec = wfrec - learning_rate2 * layer_forget_syth_rec

        wix   = wix - learning_rate * layer_input_syth_x
        wirec = wirec - learning_rate2 * layer_input_syth_rec

        wcx   = wcx - learning_rate * layer_cell_syth_x
        wcrec = wcrec - learning_rate2 * layer_cell_syth_rec

        wox   = wox - learning_rate * layer_out_syth_x
        worec = worec - learning_rate2 * layer_out_syth_rec

        wfx_syth_passon =   convolve(wfx,  np.rot90(np.pad(layer_forget_syth_part_x *   layer_forget_syth_part_2,((1,1),(1,1),(0,0)),'constant'),2),'valid')
        wfrec_syth_passon = convolve(wfrec,np.rot90(np.pad(layer_forget_syth_part_rec * layer_forget_syth_part_2,((1,1),(1,1),(0,0)),'constant'),2),'valid')

        wix_syth_passon =   convolve(wix,  np.rot90(np.pad(layer_input_syth_part_x *   layer_input_syth_part_2,((1,1),(1,1),(0,0)),'constant'),2),'valid')
        wirec_syth_passon = convolve(wirec,np.rot90(np.pad(layer_input_syth_part_rec * layer_input_syth_part_2,((1,1),(1,1),(0,0)),'constant'),2),'valid')

        wcx_syth_passon =   convolve(wcx,  np.rot90(np.pad(layer_cell_syth_part_x *   layer_cell_syth_part_2,((1,1),(1,1),(0,0)),'constant'),2),'valid')
        wcrec_syth_passon = convolve(wcrec,np.rot90(np.pad(layer_cell_syth_part_rec * layer_cell_syth_part_2,((1,1),(1,1),(0,0)),'constant'),2),'valid')

        wox_syth_passon =   convolve(wox,  np.rot90(np.pad(layer_out_syth_part_x *   layer_out_syth_part_2,((1,1),(1,1),(0,0)),'constant'),2),'valid')
        worec_syth_passon = convolve(worec,np.rot90(np.pad(layer_out_syth_part_rec * layer_out_syth_part_2,((1,1),(1,1),(0,0)),'constant'),2),'valid')

        return layer_output,(wfx_syth_passon,wfrec_syth_passon,wix_syth_passon,wirec_syth_passon,wcx_syth_passon,wcrec_syth_passon,wox_syth_passon,worec_syth_passon)

    def synthetic_weight_update(self,gradiend_from_futuer_layer = None):
        
        wfx_SGD   = self.layer_forget_syth_part_x_save   - gradiend_from_futuer_layer[0]
        wfrec_SGD = self.layer_forget_syth_part_rec_save - gradiend_from_futuer_layer[1]
        self.wfx_syth =    self.wfx_syth -   learning_rate * np.rot90( convolve(wfx_SGD,np.rot90(  np.pad(self.layer_output_save,((1,1),(1,1),(0,0)),'constant') ,2),'valid' )  ,2)
        self.wfrec_syth =  self.wfrec_syth - learning_rate2 *np.rot90( convolve(wfrec_SGD,np.rot90(np.pad(self.layer_output_save,((1,1),(1,1),(0,0)),'constant') ,2),'valid' )  ,2)

        wix_SGD   = self.layer_forget_syth_part_x_save   - gradiend_from_futuer_layer[2]
        wirec_SGD = self.layer_forget_syth_part_rec_save - gradiend_from_futuer_layer[3]
        self.wix_syth =    self.wix_syth -   learning_rate * np.rot90( convolve(wix_SGD,np.rot90(  np.pad(self.layer_output_save,((1,1),(1,1),(0,0)),'constant') ,2),'valid' )  ,2)
        self.wirec_syth =  self.wirec_syth - learning_rate2 *np.rot90( convolve(wirec_SGD,np.rot90(np.pad(self.layer_output_save,((1,1),(1,1),(0,0)),'constant') ,2),'valid' )  ,2)

        wcx_SGD   = self.layer_forget_syth_part_x_save   - gradiend_from_futuer_layer[4]
        wcrec_SGD = self.layer_forget_syth_part_rec_save - gradiend_from_futuer_layer[5]
        self.wcx_syth =    self.wcx_syth -   learning_rate * np.rot90( convolve(wcx_SGD,np.rot90(  np.pad(self.layer_output_save,((1,1),(1,1),(0,0)),'constant') ,2),'valid' )  ,2)
        self.wcrec_syth =  self.wcrec_syth - learning_rate2 *np.rot90( convolve(wcrec_SGD,np.rot90(np.pad(self.layer_output_save,((1,1),(1,1),(0,0)),'constant') ,2),'valid' )  ,2)

        wox_SGD   = self.layer_forget_syth_part_x_save   - gradiend_from_futuer_layer[6]
        worec_SGD = self.layer_forget_syth_part_rec_save - gradiend_from_futuer_layer[7]
        self.wox_syth =    self.wox_syth -   learning_rate * np.rot90( convolve(wox_SGD,np.rot90(  np.pad(self.layer_output_save,((1,1),(1,1),(0,0)),'constant') ,2),'valid' )  ,2)
        self.worec_syth =  self.worec_syth - learning_rate2 *np.rot90( convolve(worec_SGD,np.rot90(np.pad(self.layer_output_save,((1,1),(1,1),(0,0)),'constant') ,2),'valid' )  ,2)

# 3. Declare all of the weigths
layer_1 = Decoupled_LSTM_Layer(1)
layer_2 = Decoupled_LSTM_Layer(2)
layer_3 = Decoupled_LSTM_Layer(3)

# 3.5 Before Training
print('====== BEFORE TRAINING ========')
for image_index in range(0,len(test_data),3):
    
    current_label =         test_label[image_index,:]
    current_first_input_1 = test_data[image_index,:,:,:]
    hidden_state[1,:,:,:],_ =               layer_1.feed_forward_synthetic_update(current_first_input_1,hidden_state[0,:,:,:])

    current_first_input_2 = test_data[image_index+1,:,:,:]
    hidden_state[2,:,:,:],gradiend_from_2 = layer_2.feed_forward_synthetic_update(current_first_input_2,hidden_state[1,:,:,:])

    current_first_input_3 = test_data[image_index+2,:,:,:]
    hidden_state[3,:,:,:],gradiend_from_3 = layer_3.feed_forward_synthetic_update(current_first_input_3,hidden_state[2,:,:,:])

    final_layer_input = np.reshape(hidden_state[3,:,:,:],(1,-1))

    final_l1  = final_layer_input.dot(w_final_1)
    final_l1A = tanh(final_l1)

    final_l2  = final_l1A.dot(w_final_2)
    final_l2A = log(final_l2)

    print("Before Current Prediction : ", final_l2A," Current Prediction Rouned : ", np.round(final_l2A), " current Ground Truth: ", current_label)
print('=============================')


# 4. Training via the epoch
for iter in range(num_epoch):
    
    for image_index in range(0,len(train_data),3):
    
        current_label =      train_label[image_index,:]

        current_first_input_1 = train_data[image_index,:,:,:]
        hidden_state[1,:,:,:],_ =    layer_1.feed_forward_synthetic_update(current_first_input_1,hidden_state[0,:,:,:])

        current_first_input_2 = train_data[image_index+1,:,:,:]
        hidden_state[2,:,:,:],gradiend_from_2 = layer_2.feed_forward_synthetic_update(current_first_input_2,hidden_state[1,:,:,:])
        layer_1.synthetic_weight_update(gradiend_from_2)

        current_first_input_3 = train_data[image_index+2,:,:,:]
        hidden_state[3,:,:,:],gradiend_from_3 = layer_3.feed_forward_synthetic_update(current_first_input_3,hidden_state[2,:,:,:])
        layer_2.synthetic_weight_update(gradiend_from_3)

        final_layer_input = np.reshape(hidden_state[3,:,:,:],(1,-1))

        final_l1  = final_layer_input.dot(w_final_1)
        final_l1A = tanh(final_l1)

        final_l2  = final_l1A.dot(w_final_2)
        final_l2A = log(final_l2)

        cost = np.square(final_l2A - current_label).sum() * 0.5
        print("Current Batch : ",image_index, " Current Ground Truth : ", current_label ,"Real Cost update Cost : ", cost, end='\r')
        
        total_cost =+ cost

        grad_final2_part_1 = final_l2A - current_label
        grad_final2_part_2 = d_log(final_l2)
        grad_final2_part_3 = final_l1A
        grad_final2 = grad_final2_part_3.T.dot(grad_final2_part_1  * grad_final2_part_2)

        grad_final1_part_1 = (grad_final2_part_1 * grad_final2_part_2).dot(w_final_2.T)
        grad_final1_part_2 = d_tanh(final_l1)
        grad_final1_part_3 = final_layer_input
        grad_final1 = grad_final1_part_3.T.dot(grad_final1_part_1  * grad_final1_part_2)

        w_final_2 = w_final_2 - lr3 * grad_final2
        w_final_1 = w_final_1 - lr3 * grad_final1
        

        grad_layer_3_part_1 = np.reshape((grad_final1_part_1 * grad_final1_part_2).dot(w_final_1.T),(100,100,3))
        layer_3.synthetic_weight_update((grad_layer_3_part_1,grad_layer_3_part_1,grad_layer_3_part_1,grad_layer_3_part_1,grad_layer_3_part_1,grad_layer_3_part_1,grad_layer_3_part_1,grad_layer_3_part_1))

    if iter % 2 == 0:
        print(" Current Iter: ", iter," Current iter :", iter, " Current batch: ", image_index, " Current Total Cost : ", total_cost, end='\n')
    
    cost_array.append(total_cost)
    total_cost = 0

print('\n====== AFTER TRAINING ========')
for image_index in range(0,len(test_data),3):
    
    current_label =         test_label[image_index,:]
    current_first_input_1 = test_data[image_index,:,:,:]
    hidden_state[1,:,:,:],_ =               layer_1.feed_forward_synthetic_update(current_first_input_1,hidden_state[0,:,:,:])

    current_first_input_2 = test_data[image_index+1,:,:,:]
    hidden_state[2,:,:,:],gradiend_from_2 = layer_2.feed_forward_synthetic_update(current_first_input_2,hidden_state[1,:,:,:])

    current_first_input_3 = test_data[image_index+2,:,:,:]
    hidden_state[3,:,:,:],gradiend_from_3 = layer_3.feed_forward_synthetic_update(current_first_input_3,hidden_state[2,:,:,:])

    final_layer_input = np.reshape(hidden_state[3,:,:,:],(1,-1))

    final_l1  = final_layer_input.dot(w_final_1)
    final_l1A = tanh(final_l1)

    final_l2  = final_l1A.dot(w_final_2)
    final_l2A = log(final_l2)

    print("Current Prediction : ", final_l2A," Current Prediction Rouned : ", np.round(final_l2A), " current Ground Truth: ", current_label)
print('=============================')

plt.title("Cost over time")
plt.plot(np.arange(len(cost_array)), cost_array)
plt.show()











# -- end code --