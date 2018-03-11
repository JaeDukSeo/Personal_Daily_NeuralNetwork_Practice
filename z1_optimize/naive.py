
def naive(x,y):

    z = 0 
    while x>0:
        z= z +y
        x = x-1
    return z

print(naive(7,3))