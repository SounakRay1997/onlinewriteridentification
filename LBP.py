import numpy as np

def distance(neighbor,point):
    d=(neighbor[2]-point[2])**2+(neighbor[3]-point[3])**2
    return d
    
    
def descriptor(substroke):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    encoding=[]
    for point in substroke:
        neighbors=[]
        distances=[]
        for neighbor in substroke:
            if neighbor!=point:
                neighbors.append(neighbor)
                distances.append(distance(neighbor,point))
        #neighbors.sort(key=distance)
        for i,neighbor in enumerate(neighbors):
            if(neighbor[4]>point[4]):
                neighbors[i]=1
            else:
                neighbors[i]=0
        Z=[x for _, x in sorted(zip(distances, neighbors),reverse=True)]
        i=0
        p=0
        for neighbor in Z:
            if(neighbor):
                p+=2**i
            i+=1
        encoding.append(p)    
    return encoding
        # return the histogram of Local Binary Patterns

        #return hist
        
s=[[0, 0, 1, 2, 3], [0, 0, 2, 3, 4], [0, 0, 4, 5, 6]]
a=descriptor(s)
print(a)
