import numpy as np
import queue
import time
import sys
sys.setrecursionlimit(2000)


# Run the Program Multiple Times Just to see
for iter in range(5):
    # 0. Crate Matrix size n
    n = 15
    matrix = np.zeros((n,n))

    # 1. Generate Random X and Y coordinates to put 1
    x = np.random.randint(n)
    y = np.random.randint(n)
    matrix[y,x] = 1

    # 2. We are going to start at zero zero 
    start_x,start_y = np.random.randint(n),np.random.randint(n)

    # =======================================
    # Implement BFS
    def BFS(queue=None):
        
        current_index = queue.get()
        current_x,current_y = current_index[0],current_index[1]

        element = matrix[current_y,current_x]

        if element == 1: return current_x,current_y

        for n in range(current_x-1,current_x+2):
            for m in range(current_y-1,current_y+2):
                if not (n==current_x and m==current_y) \
                    and n>-1 and m>-1 \
                    and n<matrix.shape[0] and m<matrix.shape[1] \
                    and (n,m) not in queue.queue :
                        queue.put((n,m))    
        return BFS(queue)

    # Pure Loop DFS
    def DFS_Pure_Loop():
        element = None
        for n in range(matrix.shape[0]):
            for m in range(matrix.shape[0]):
                element = matrix[m,n]
                if element == 1: return n,m


    # Implement DFS - starting from anywhere but reccursion depth is problem 
    def DFS(current_x=None,current_y=None,visited=None):
        
        visited.append((current_x,current_y))
        element = matrix[current_y,current_x]

        if element == 1: return current_x,current_y

        if current_x<0:return
        if current_x>=matrix.shape[0]: return
            
        if current_y<0:return
        if current_y>=matrix.shape[1]: return

        for n in range(current_x-1,current_x+2):
            for m in range(current_y-1,current_y+2):
                if not (n==current_x and m==current_y) \
                    and n>-1 and m>-1 \
                    and n<matrix.shape[0] and m<matrix.shape[1] \
                    and (n,m) not in visited:
                        return DFS(n,m,visited)
    # =======================================

    # 3. Found by DFS
    visited = []
    DFSstart = time.time()
    DFS_results = DFS(start_x,start_y,visited)
    DFSend = time.time()
    # DFS_results = DFS()

    # 4. Queue for BFS
    start_queue = queue.Queue()
    start_queue.put((start_x,start_y))
    BFSstart = time.time()
    BFS_results = BFS(start_queue)
    BFSend = time.time()


    # Print out the statements
    print('======== Given Matrix ========')
    print(matrix)

    print('======== Given Starting Coord ========')
    print("Starting X: ",start_x," Starting Y: ",start_y)

    print('======== Given Answers ========')
    print("Solution by DFS: ",DFS_results, " Execution Time : ", DFSend-DFSstart)
    print("Solution by BFS: ",BFS_results, " Execution Time : ", BFSend-BFSstart)



# -- end code --