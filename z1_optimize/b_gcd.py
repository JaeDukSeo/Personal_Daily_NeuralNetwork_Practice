import numpy as np
import time
def gcd_1(a,b):
    best = 0

    for i in range(1,a+b):
        if a%i==0 and b%i==0:
            best = i
    return best

def gcd_2(a,b):
    if b == 0 : return a

    a_prime = a % b
    return    gcd_2(b,a_prime) 

start = time.time()

print(gcd_1(17890768,17890768))

end = time.time()
print(end - start)

start = time.time()

print(gcd_2(357,234))

end = time.time()
print(end - start)


