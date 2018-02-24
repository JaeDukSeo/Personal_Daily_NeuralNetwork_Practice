import numpy as np
import time 

global m,n,surface

m = 5
n = 5
surface = np.zeros((m,n))

def spill(x,y,stren):

        global m,n,surface

        if stren == 0 : return
        if x < 0 or x >= m or y <0 or y>=n: return

        cell = surface[x,y]

        if cell == -1 :return 



print('done') 


# -- end code --