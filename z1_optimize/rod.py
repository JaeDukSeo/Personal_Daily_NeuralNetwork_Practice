import numpy as np
import sys

price = np.array([4,15,29,9,10,127,17,70])

def bestprice(n):
    if n<=0: return 0   

    # result_array is an array with length n and initialized with all 0 values 
    result_array = [0] * (n)

    # Outer Loop to Calculate the best value at n
    for i in range(0,n):
        Best_price_until_current_position  = 0

        # Inner Loop to Calcaulte the best value until n
        for j in range(i+1):
            Best_price_until_current_position  = \
                    max(Best_price_until_current_position , \
                        price[j]+result_array[i-j-1])

        # Store the best value until n 
        result_array[i] = Best_price_until_current_position 

    # Since we are assuming that array index starts from 0 return element 
    # at n-1
    return result_array[n-1]


print("Max Price at length 0: ",bestprice(0))
print("Max Price at length 1: ",bestprice(1))
print("Max Price at length 2: ",bestprice(2))
print("Max Price at length 3: ",bestprice(3))
print("Max Price at length 4: ",bestprice(4))
print("Max Price at length 5: ",bestprice(5))
print("Max Price at length 6: ",bestprice(6))
print("Max Price at length 7: ",bestprice(7))
print("Max Price at length 8: ",bestprice(8))


























sys.exit()
INT_MIN = -9999999999999
# Returns the best obtainable price for a rod of length n and
# price[] as prices of different pieces
def cutRod(price, n):
    val = [0 for x in range(n+1)]
    val[0] = 0

    # Build the table val[] in bottom up manner and return the last entry from the table

    # For N Time - 1 2 3 
    for i in range(1, n+1):
        max_val = INT_MIN

        # The triangle approach to the problem
        for j in range(i):
            max_val = max(max_val, price[j] + val[i-j-1])

        # Max Price Until i approach
        val[i] = max_val
 
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