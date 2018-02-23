
import numpy as np

large_l = list(range(10) ) + np.random.randint(10)
finish = []
temp = []

lenss = len(large_l)
for iter in range(lenss):
    
    large = max(large_l)
    finish.append(large)
    ss = large_l.index(large)
    large_l.pop(ss)
    

print(finish)


# -- end code --