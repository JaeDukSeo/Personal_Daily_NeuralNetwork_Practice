
import time

start = time.time()

def fib(x):

    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return  fib(x-1) + fib(x-2)


def fiib(x):
    if x <= 1:
        return x
    else:
        return fiib(x-1) + fiib(x-2)
start = time.time()
print(fib(4))
end = time.time()
print(end - start)
start = time.time()


import numpy as np 
temp = [None] * 1801
temp[0] = 0
temp[1] = 1

for ii in range(2,1801):
    temp[ii] = temp[ii-1] + temp[ii-2]

print(temp[-1])
end = time.time()
print(end - start)

