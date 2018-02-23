


def minre(x,n,l):
    

    numfi = 0
    current = 0

    while current <= n:
        
        lastrefil = current

        while current <= n and x[current+1] - x[lastrefil] <= l:
            current = current + 1
        
        if current == lastrefil: return "death"

        if current <= n:
            numfi = numfi + 1

    return numfi

x = [1,4,5,3,9]


print(minre(x,len(x),4))

# -- end code --