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




# -- end code --