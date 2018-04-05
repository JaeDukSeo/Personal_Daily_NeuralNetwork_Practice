import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu
from numpy.linalg import inv

def bubble_sort(array):
    for x in range(len(array)-1):
        for xx in range(x,len(array)):
            if array[x] > array[xx]:
                temp = array[x]
                array[x] = array[xx]
                array[xx] = temp
    return array 

one = np.array([3,4,2,5,43])
two = np.array([2,3,488,291])
new_array = np.concatenate((one,two))
# print(new_array)
# print(bubble_sort(new_array))



time = np.array([0,2,4,6,8,10,12])
pppl = np.array([0,350,1100,2400,6500,8850,10000])
total_30_percent = pppl[-1] * 0.3
for x in range(len(pppl)):
    if total_30_percent<pppl[x]:
        before = time[x-1]
        after = time[x]
        # print('Between :',before, ' and ',after)
        # print("Average : ", (before+after)/2)
        break


data_point_raw = np.array([1,2,3,4,7777,5,6,9,2,3])
plt.plot(data_point_raw)
plt.title("Original")
# plt.show()

data_normal = (data_point_raw - data_point_raw.min())/(data_point_raw.max()- data_point_raw.min())
plt.title("Normalized")
plt.plot(data_normal)
# plt.show()

mean = data_point_raw.sum(axis=0) /len(data_point_raw)
var = ((data_point_raw-mean) ** 2).sum(axis=0) / len(data_point_raw)
std_x = (data_point_raw-mean)/( (var + 1e-8) ** 0.5 )
plt.title("Std")
plt.plot(std_x)
# plt.show()


temp_mat = np.array([
    [2,10],
    [-1,2]
])

for_math = np.array([
    [2,10],
    [1,2]
])
solved_naive = (1/(temp_mat[0,0] * temp_mat[1,1] - temp_mat[0,1]*temp_mat[1,0])) * for_math

print('Naive Solution : \n',solved_naive)
print('---------')
print('Library Solution :\n',inv(temp_mat))



# -- end code --