import time
class MyQUEUE:
        #--------------- implementation of a queue---------------
	
	def __init__(self):
		self.store = []
		
	def enqueue(self,val):
		self.store.append(val)
		
	def dequeue(self):
		val = None
		try:
			val = self.store[0]
			if len(self.store) == 1:
				self.store = []
			else:
				self.store = self.store[1:]	
		except:
			pass
			
		return val	
		
	def IsEmpty(self):
		result = False
		if len(self.store) == 0:
			result = True
		return result
path_queue = MyQUEUE() # creating a queue queue

#--------------------BREADTH FIRST SEARCH---------------------
def BFS(graph,start,end,q):
	temp_path = [start]
	q.enqueue(temp_path)
	while q.IsEmpty() == False:
		tmp_path = q.dequeue()
		last_node = tmp_path[len(tmp_path)-1]
		#print tmp_path
		if last_node == end:
			print( "VALID_PATH : ",tmp_path)
		for link_node in graph[last_node]:
			if link_node not in tmp_path:
				new_path = []
				new_path = tmp_path + [link_node]
				q.enqueue(new_path)
def matrix():
    o = input('Enter order of the matrix ')
    m = [[]]
    #------------------Matrix initialization-------------------
    m = [[0 for elem in range(o)] for elem1 in range(o)]
    time.sleep(1)
    print('-----------------Enter elements in the matrix------------------- ')
    #-----------------------Matrix input-----------------------
    for elem in range(o):
	    print(' ENTER ELEMENTS FOR row ! ' ,elem+1)
        for elem1 in range(o):
            m[elem][elem1] = input('')
    #----------------------display Matrix----------------------
    print('--------------------YOUR MATRIX ! ------------------')
    time.sleep(2)
    for elem in range(o):
        for elem1 in range(o):
            print(m[elem][elem1])
            print(' ')
        print('')
    graph = {}
    l = []
    #-----------------------creating Graph---------------------
    #-------------------Converting matrix to graph-------------
    for elem in range(o):
        
        for elem1 in range(o):
             l = []
             if m[elem][elem1] == 1:
                 # below the element
                 if (elem + 1) <= (o - 1):
                     if m[elem + 1][elem1] == 1:
                         l.append(str(elem + 1) + str(elem1))
                 #above the element
                 if (elem - 1) >= 0:
                     if m[elem - 1][elem1] == 1:
                         l.append(str(elem - 1) + str(elem1))
                 # right of the element
                 if (elem1 + 1) <= (o - 1):
                     if m[elem][elem1 + 1] == 1:
                         l.append(str(elem) + str(elem1 + 1))
                 # left of the element
                 if (elem1 - 1) >= 0:
                     if m[elem][elem1 - 1] == 1:
                         l.append(str(elem) + str(elem1 - 1))
                 graph[str(elem) + str(elem1)] = l
             else:
                 graph[str(elem) + str(elem1)] = l
    print('------------GRAPH OBTAINED FROM MATRIX : -------------')
    time.sleep(2)
    print(graph)
    return graph
graph = matrix()
start = raw_input("Enter start index in single number eg: 23 for index (2,3) ")
end = raw_input("Enter end index in single number eg: 23 for index(2,3) ")
time.sleep(2)
print '------------PRINTING VALID PATHS ! ---------------'
time.sleep(2)
print '-----------PROGRAM ENDS IF TRAVERSAL IS NOT POSSIBLE------------'
BFS(graph,start,end,path_queue)