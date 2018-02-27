import numpy as np
import time,sys
from collections import deque
from collections import defaultdict
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

# Use the edge
# void spillBFS(int x,int y, int strength){
#       
#       if strength =0 return 
#       if x<0 or x>=m or y<0 or y>=n return
#       int cell = surface(x,y)
#       if cell is an OBSTACLE   return
#       surface(x,y) = strength
# 
#       queue = new Queue()
# 
# }

# def spillBFS(int lefttop,int leftbottom,int rightop, int rightbottom):
# 
# 
#



# HAVE TO do this
def spillBFS( x_left,  x_right, y_top, y_bottom,strength):

    global m,n,surface
    print('Current Strength: ',strength, ' Current X Small: ', x_left, x_right, y_top, y_bottom)

    if strength == 0: return
    if x_left == -1: x_left = 0
    if x_right == n-1: x_right = n-2
    if y_top == -1: y_top = 0
    if y_bottom == m-1: y_bottom = m-2

    if x_left==x_right and y_top == y_bottom:
        if x_left<0 or x_left >n-1 or y_top<0 or y_top>m-1: return
        if not surface[y_top,x_left] == -1 :
            surface[y_top,x_left] = strength
        spillBFS(x_left-1,x_right+1,y_top-1,y_bottom+1,strength-1)
    else:
        
        print('---- Going Across Top----')
        for x_cor in range(x_left,x_right+1):
            print(y_top,x_cor)
            if not surface[y_top,x_cor] == -1 and surface[y_top,x_cor]<strength:
                surface[y_top,x_cor] = strength
        print('---- Going Across Top----')

        print('---- Going Across Bottom----')
        for x_cor in range(x_left,x_right+1):
            print(y_bottom,x_cor)
            if not surface[y_bottom,x_cor] == -1 and surface[y_bottom,x_cor]<strength:
                surface[y_bottom,x_cor] = strength
        print('---- Going Across Bottom----')

        print('---- Going Down left----')
        for y_cor in range(y_top+1,y_bottom):
            print(y_cor,x_left)
            if not surface[y_cor,x_left] == -1 and surface[y_cor,x_left]<strength:
                surface[y_cor,x_left] = strength
        print('---- Going Down left----')
        
        print('---- Going Down right----')
        for y_cor in range(y_top+1,y_bottom):
            print(y_cor,x_right)
            if not surface[y_cor,x_right] == -1 and surface[y_cor,x_right]<strength:
                surface[y_cor,x_right] = strength
        print('---- Going Down right----\n')
        spillBFS(x_left-1,x_right+1,y_top-1,y_bottom+1,strength-1)

m = 7  
n = 7
surface = np.zeros((m,n))
start_time = time.time()

# surface[4,4] = -1
surface[0,1] = -1
surface[1,1] = -1
# surface[1,0] = -1
surface[2,1] = -1

surface[m-1,n-1] = -1

x_coordinate_spill = 0
y_coordinate_spill = 0
strength = 4
print("Gloabal: y ",m, ' and x ', n)
print(surface)
spillBFS(x_coordinate_spill,x_coordinate_spill,y_coordinate_spill,y_coordinate_spill,strength)
print("--- %s seconds ---" % (time.time() - start_time))
print(surface)


sys.exit()

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


 
# This class represents a directed graph using adjacency
# list representation
class Graph:
 
    # Constructor
    def __init__(self):
 
        # default dictionary to store graph
        self.graph = defaultdict(list)
 
    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    # Function to print a BFS of graph
    def BFS(self, s):
 
        # Mark all the vertices as not visited
        visited = [False]*(len(self.graph))
 
        # Create a queue for BFS
        queue = []
 
        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True
 
        while queue:
 
            # Dequeue a vertex from queue and print it
            s = queue.pop(0)
            print(s+1,'here')
 
            # Get all adjacent vertices of the dequeued
            # vertex s. If a adjacent has not been visited,
            # then mark it visited and enqueue it
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True



# Driver code
# Create a graph given in the above diagram
g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
 
print("Following is Breadth First Traversal (starting from vertex 2)")
g.BFS(2)
 




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