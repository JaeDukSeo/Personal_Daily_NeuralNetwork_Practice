import numpy as np


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
print(new_array)
print(bubble_sort(new_array))





# -- end code --