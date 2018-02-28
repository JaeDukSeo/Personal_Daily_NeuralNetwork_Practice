import numpy as np
import time,sys
from queue import *


def spill_BFS_final(x,y,strength,passon_queue,strength_queue):
    global m,n,surface

    if strength == 0: return
    if x<0 or y<0: return
    if x>m-1 or y>n-1: return
    if surface[x,y] == -1: return

    surface[x,y] = strength
    
    for x_cor in range(x-1,x+2):
        for y_cor in range(y-1,y+2):
            if  (x_cor>=0 and y_cor>=0) and (x_cor<m and y_cor<n) :
                if (not(x_cor==x and y_cor==y)) and (not surface[x_cor,y_cor] == -1) and \
                    (surface[x_cor,y_cor]<strength-1) and (not (x_cor,y_cor) in passon_queue.queue):
                    passon_queue.put((x_cor,y_cor))
                    strength_queue.put(strength-1)

    if passon_queue.empty() : return
    next_coor = passon_queue.get()
    next_strength = strength_queue.get()
    spill_BFS_final(next_coor[0],next_coor[1],next_strength,passon_queue,strength_queue)

global m,n,surface

m = 9
n = 9
surface = np.zeros((m,n))
start_time = time.time()
queue = Queue()
strength_queue = Queue()

surface[0,2] = -1
surface[0,3] = -1
surface[2,6] = -1
surface[3,6] = -1
surface[3,7] = -1
surface[4,7] = -1
surface[5,7] = -1
surface[6,7] = -1
surface[5,1] = -1

surface[5,0] = -1
surface[5,1] = -1
surface[6,0] = -1
surface[6,1] = -1
surface[7,0] = -1
surface[7,1] = -1



x_coordinate_spill = 1
y_coordinate_spill = 5
strength = 5


print('======== Before Spilling Surface ==============')
print(surface)

start_time = time.time()
spill_BFS_final(x_coordinate_spill,y_coordinate_spill,strength,queue,strength_queue)
print("\n\n--- Execution Time:  %s seconds ---\n\n" % (time.time() - start_time))

print('======== FINAL Surface ==============')
print(surface)