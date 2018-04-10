import numpy as np,sys
import matplotlib.pyplot as plt
import cv2

np.random.seed(678)
np.set_printoptions(precision=0)

# generate data / label
data = np.ones((10,100))
data[:10,10:20] = 0

label = data.copy()
label[:10,:10] = 0
label[:10,30:40] = 0
label[:10,50:60] = 0
label[:10,80:90] = 0

# hyper 
q_table = np.ones((10))
epilson = 0
num_epid = 8000
 
learing_rate = 0.8
discount  = 0.006

# train
for epid in range(num_epid):
    
    for state in range(0,label.shape[1]-10,10):

        current_state = data[:10,state:state+10]
        current_label = label[:10,state:state+10]
        
        epilson = np.random.randint(10)
        if epilson < 5:
            action = q_table[int(state/10)]
        else: 
            action = q_table[int(state/10)] - np.random.randn()/(epid+1e-8)

        ret, create_data = cv2.threshold(current_state, action, 1, cv2.THRESH_BINARY)
        reward = (1.0 -  cv2.bitwise_xor(current_label,create_data)).sum() / 100.0

        plt.imshow(current_label,cmap='gray')
        plt.show()

        q_table[int(state/10)] = q_table[int(state/10)] + learing_rate * (
            reward + discount * q_table[int(state/10)+1] - q_table[int(state/10)] 
        )
