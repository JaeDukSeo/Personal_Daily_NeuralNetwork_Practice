
import time

start = time.time()

def fib(x):

    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return x + fib(x-1)


def fiib(x):
    if x <= 1:
        return x
    else:
        return fiib(x-1) + fiib(x-2)
start = time.time()
print(fiib(40))
end = time.time()
print(end - start)