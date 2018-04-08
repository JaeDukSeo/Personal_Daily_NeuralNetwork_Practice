import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 
df = pd.read_csv("SampleCSVFile_11kb.csv",encoding="latin1")
added = df.ix[:,0] + np.random.randn(99)* 20.5

print("Original Mean :",df.ix[:,0].mean())
print("Original std :",df.ix[:,0].std())
print("Original var :",df.ix[:,0].var())
print('=========================')
print("manipulate Mean :",added.mean())
print("manipulate std :",added.std())
print("manipulate var :",added.var())

plt.plot(range(len(added)),added,label='Added Noise')
plt.plot(range(len(added)),df.ix[:,0],label='Original')
plt.legend()
plt.show()
print('=========================')


# 2. 
value1 = np.random.randn(1000) * 3.6 + 3
value2 = np.random.randn(1000) * 1.7
bins = np.linspace(-10, 10, 100)
plt.hist(value1,bins=bins,label='one',color ='r',alpha=0.5)
plt.hist(value2,bins=bins,label='two',color ='b',alpha=0.5)
plt.legend()
plt.show()
print('=========================')


# 3. 
some_data = [2,3,-8,2,-0.8,93,222,102,3,1,333,82,9] 

def insert_sort(input_list):
    
    for i in range(1,len(input_list)):
        
        current = input_list[i]
        hole = i

        while hole > 0 and input_list[hole-1]>current:
            input_list[hole] = input_list[hole-1]
            hole = hole - 1
        
        input_list[hole] = current

    return input_list

print(insert_sort(some_data))



# -- end code --