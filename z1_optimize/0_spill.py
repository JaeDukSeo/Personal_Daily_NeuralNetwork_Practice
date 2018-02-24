import numpy as np
import time 
from collections import deque

global m,n,surface

m = 13  
n = 13
surface = np.zeros((m,n))

# 0. Original Spill
def spill(x,y,stren):

        global m,n,surface

        if stren == 0 : return
        if x < 0 or x >= m or y <0 or y>=n: return

        cell = surface[x,y]

        if cell == -1 : return 
        if cell < stren :
            surface[x,y] = stren
        
        spill(x-1,y-1,stren-1)
        spill(x-1,y,stren-1)
        spill(x-1,y+1,stren-1)
        spill(x,y-1,stren-1)
        spill(x,y+1,stren-1)
        spill(x+1,y-1,stren-1)
        spill(x+1,y,stren-1)
        spill(x+1,y+1,stren-1)

# 1. Spill Short cut
def spill2(x,y,stren):
    
        global m,n,surface

        if stren == 0 : return
        if x < 0 or x >= m or y <0 or y>=n: return

        cell = surface[x,y]

        if cell == -1 : return 
        if cell >= stren : return
        surface[x,y] = stren
        
        spill(x-1,y-1,stren-1)
        spill(x-1,y,stren-1)
        spill(x-1,y+1,stren-1)
        spill(x,y-1,stren-1)
        spill(x,y+1,stren-1)
        spill(x+1,y-1,stren-1)
        spill(x+1,y,stren-1)
        spill(x+1,y+1,stren-1)



# This is really slow, since the strength is also in the way. 
# It takes 6 seconds lol
m = 7  
n = 7
surface = np.zeros((m,n))
start_time = time.time()
spill(7,7,8)
print("--- %s seconds ---" % (time.time() - start_time))

# Short cut actually make it worse LOL
m = 7  
n = 7
surface = np.zeros((m,n))
start_time = time.time()
spill2(7,7,8)
print("--- %s seconds ---" % (time.time() - start_time))

# Depth first search








#Creating a Queue
queue = deque([1,5,8,9])

#Enqueuing elements to the Queue
queue.append(7) #[1,5,8,9,7]
queue.append(0) #[1,5,8,9,7,0]

#Dequeuing elements from the Queue
queue.popleft() #[5,8,9,7,0]
queue.popleft() #[8,7,9,0]

#Printing the elements of the Queue
print(queue)

# -- end code -------