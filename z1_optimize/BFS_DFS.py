import numpy as np




# 0. Crate Matrix size n
n = 9
matrix = np.zeros((n,n))
print(matrix.shape)
print(matrix)

# 1. Generate Random X and Y coordinates
x = np.random.randint(n)
y = np.random.randint(n)

matrix[y,x] = 1
print("Generated X : ",x, ' Generated Y: ',y)
print(matrix)



# -- end code --