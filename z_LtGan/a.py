import numpy as np

m = 3
n = 3
matrix = np.zeros((m,n))

print(matrix)
print(matrix[0,0])

global count 
count = 0
def spill(x,y,stre):
    global count 

    if stre == 0: return
    if x<-1 or x>=m or y<-1 or y>=n: return 
    cell= matrix[x,y]
    matrix[x,y] = cell + 1
    count = count + 1
    if cell == -1 : return
    # if cell < stre : matrix[x,y] = cell + 1

    spill(x-1,y-1,stre-1)
    spill(x-1,y,stre-1)
    spill(x-1,y+1,stre-1)

    spill(x,y-1,stre-1)
    spill(x,y+1,stre-1)

    spill(x+1,y-1,stre-1)
    spill(x+1,y,stre-1)
    spill(x+1,y+1,stre-1)
    
    

spill(1,1,3)

print(matrix)
print(count)

