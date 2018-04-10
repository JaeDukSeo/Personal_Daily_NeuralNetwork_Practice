import numpy as np
import matplotlib.pyplot as plt
import cv2
data = np.ones((64,64))
data[:16,:16] = 0

ground_truth = data.copy()
ground_truth[48:,48:] = 0
ground_truth[48:,32:48] = 0
ground_truth[:16,32:48] = 0
ground_truth[48:,:16] = 0

M = data.shape[0]//4
N = data.shape[1]//4
data_tiles = [data[x:x+M,y:y+N] for x in range(0,data.shape[0],M) for y in range(0,data.shape[1],N)]
label_tiles = [ground_truth[x:x+M,y:y+N] for x in range(0,ground_truth.shape[0],M) for y in range(0,ground_truth.shape[1],N)]

# -- Hyper --
q_table = np.random.randn(16)
q_table = np.zeros((16))

num_epid = 6000

learing_rate = 0.7
discount = 0.1

# train
for epid in range(num_epid):
    
    for step in range(len(data_tiles)-1):
        
        current_data = data_tiles[step]
        action = q_table[step] + np.random.randn()
        label_data = label_tiles[step]
        ret, create_data = cv2.threshold(label_data, action, 1, cv2.THRESH_BINARY)
        reward = (1.0 -  cv2.bitwise_xor(label_data,create_data)).sum()
        q_table[step] = q_table[step]  + learing_rate* (reward + discount * q_table[step+1]-q_table[step])


# test
temp = ""
temp2 = ""
temp3 = ""

for i in range(len(data_tiles)):
    
    action_threshold = q_table[i] 
    current_data = data_tiles[i]
    label_data = label_tiles[i]
    
    ret, results = cv2.threshold(current_data, action_threshold, 1, cv2.THRESH_BINARY)

    if i == 0:
        temp = results
        temp2 = current_data
        temp3 = label_data
    else:
        temp = np.hstack((temp,results))
        temp2 = np.hstack((temp2,current_data))
        temp3 = np.hstack((temp3,label_data))
        

plt.imshow(temp,cmap='gray')
plt.show()

plt.imshow(temp2,cmap='gray')
plt.show()

plt.imshow(temp3,cmap='gray')
plt.show()



# -- end code --