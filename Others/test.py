# Max of 3
def max_2(a,b):
    if a > b:
        return a
    else:
        return b    
        
def max3(a,b,c):
    return max_2(a,max_2(b,c))


print(max3(3,9,6))


#max of 3

def max3