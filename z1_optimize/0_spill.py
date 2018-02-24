import numpy as np
import time 

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


# This is really slow, since the strength is also in the way. 
# It takes 6 seconds lol
print(surface)
start_time = time.time()
spill(7,7,8)
print("--- %s seconds ---" % (time.time() - start_time))
print(surface)




# -- end code -------