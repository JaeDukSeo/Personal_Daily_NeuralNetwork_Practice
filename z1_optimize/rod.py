import numpy as np
import sys

price = np.array([1,5,8,9,10,17,17,20])
print(price.shape)






INT_MIN = -9999999999999
# Returns the best obtainable price for a rod of length n and
# price[] as prices of different pieces
def cutRod(price, n):
    val = [0 for x in range(n+1)]
    val[0] = 0
    print(val)
 
    # Build the table val[] in bottom up manner and return
    # the last entry from the table
    for i in range(1, n+1):
        max_val = INT_MIN
        for j in range(i):
             max_val = max(max_val, price[j] + val[i-j-1])
        val[i] = max_val
        print(val)
 
    return val[n]



print(cutRod(price,0))
print(cutRod(price,1))
print(cutRod(price,2))
print(cutRod(price,3))
print(cutRod(price,4))
print(cutRod(price,5))
print(cutRod(price,6))
print(cutRod(price,7))
print(cutRod(price,8))



sys.exit()









global howmans
howmans= 0 
def bestprice(n):
    global howmans
    
    if n<=0: return 0   

    result = price[n-1]
    howmans = howmans + 1
    print("executed :",howmans)
    
    for length in range(0,n-1):
        ttry = price[length] 
        howmans = howmans + 1
        print("executed :",howmans)
        ttry = ttry + bestprice(n-length-1)
        if ttry > result:
            result = ttry
    return result

# print(bestprice(0))
# print(bestprice(1))
# print(bestprice(2))
# print(bestprice(3))
# print(bestprice(4))
# print(bestprice(5))
# print(bestprice(6))
print(bestprice(7))
# print(bestprice(8))

sys.exit()

# A Naive recursive solution 
# for Rod cutting problem
import sys
 
# A utility function to get the
# maximum of two integers
def max(a, b):
    return a if (a > b) else b
     
# Returns the best obtainable price for a rod of length n 
# and price[] as prices of different pieces
def cutRod(pricess, n):
    if(n <= 0):
        return 0
    max_val = -sys.maxsize-1
     
    # Recursively cut the rod in different pieces  
    # and compare different configurations
    for i in range(0, n):
        max_val = max(max_val, pricess[i] +
                      cutRod(pricess, n - i - 1))
    return max_val
 
# Driver code
# arr = [1, 5, 8, 9, 10, 17, 17, 20]
size = len(price)
print("Maximum Obtainable Value is", cutRod(price, 0))
print("Maximum Obtainable Value is", cutRod(price, 1))
print("Maximum Obtainable Value is", cutRod(price, 2))
print("Maximum Obtainable Value is", cutRod(price, 3))
print("Maximum Obtainable Value is", cutRod(price, 4))
print("Maximum Obtainable Value is", cutRod(price, 5))
print("Maximum Obtainable Value is", cutRod(price, 6))
print("Maximum Obtainable Value is", cutRod(price, 7))
print("Maximum Obtainable Value is", cutRod(price, 8))




# -- end code --