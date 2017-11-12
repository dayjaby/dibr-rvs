import numpy as np

g = 4

def f(i,x,y,arr):
    arr[x,y] = i
    i = i+1
    if i==arr.shape[0]*arr.shape[1]+1:
        result = arr.copy()
        costs = 0
        for ix in range(arr.shape[0]):
            for iy in range(arr.shape[1]):
                for jx in range(arr.shape[0]):
                    for jy in range(arr.shape[1]):
                        d = arr[ix,iy]-arr[jx,jy]
                        if d!=0:
                            costs += ((ix-jx)*(ix-jx)+g*g*(iy-jy)*(iy-jy))/(d*d)
        arr[x,y] = 0
        return costs, result
    else:
        minimalPath = None
        minimalCosts = None
        if x>0 and arr[x-1,y] == 0:
            costs, path = f(i,x-1,y,arr)
            if minimalPath is None or (costs<minimalCosts and minimalPath is not None):
                minimalPath = path
                minimalCosts = costs
        if y>0 and arr[x,y-1] == 0:
            costs, path = f(i,x,y-1,arr)
            if minimalPath is None or (costs<minimalCosts and minimalPath is not None):
                minimalPath = path
                minimalCosts = costs
        if x<arr.shape[0]-1 and arr[x+1,y] == 0:
            costs, path = f(i,x+1,y,arr)
            if minimalPath is None or (costs<minimalCosts and minimalPath is not None):
                minimalPath = path
                minimalCosts = costs
        if y<arr.shape[1]-1 and arr[x,y+1] == 0:
            costs, path = f(i,x,y+1,arr)
            if minimalPath is None or (costs<minimalCosts and minimalPath is not None):
                minimalPath = path
                minimalCosts = costs
        arr[x,y] = 0
        return minimalCosts, minimalPath

minimalCosts, minimalPath = f(1,0,0,np.zeros((5,5)))
print(minimalCosts)
print(minimalPath)
