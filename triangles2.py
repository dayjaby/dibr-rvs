import numpy as np

import pycosat
import time
import math
import itertools

"""
import theano.tensor as T
import theano
A = T.imatrix()
B = T.imatrix()
M1 = T.as_tensor_variable([A[0],A[1],B[0],B[1]])
M2 = T.as_tensor_variable([A[0],A[2],B[0],B[1]])
M3 = T.as_tensor_variable([A[1],A[2],B[0],B[1]])
M4 = T.as_tensor_variable([A[0],A[1],B[0],B[2]])
M5 = T.as_tensor_variable([A[0],A[2],B[0],B[2]])
M6 = T.as_tensor_variable([A[1],A[2],B[0],B[2]])
M7 = T.as_tensor_variable([A[0],A[1],B[1],B[2]])
M8 = T.as_tensor_variable([A[0],A[2],B[1],B[2]])
M9 = T.as_tensor_variable([A[1],A[2],B[1],B[2]])
M = T.as_tensor_variable([M1,M2,M3,M4,M5,M6,M7,M8,M9])
#M = T.ltensor3()
q  = 1.0*(M[:,3,1]-M[:,2,1])*(M[:,1,0]-M[:,0,0])-(M[:,3,0]-M[:,2,0])*(M[:,1,1]-M[:,0,1])
ua = (M[:,3,0]-M[:,2,0])*(M[:,0,1]-M[:,2,1])-(M[:,3,1]-M[:,2,1])*(M[:,0,0]-M[:,2,0])
ub = (M[:,1,0]-M[:,0,0])*(M[:,0,1]-M[:,2,1])-(M[:,1,1]-M[:,0,1])*(M[:,0,0]-M[:,2,0])
intersct = T.and_(T.neq(q,0),T.and_(0<ua/q,T.and_(0<ub/q,T.and_(1>ua/q,1>ub/q))))
triangle_intersect = theano.function([A,B],T.elemwise.Any()(intersct),allow_input_downcast=True)"""

def intersect(A,B,C,D):
    q  = 1.0*(D[1]-C[1])*(B[0]-A[0])-(D[0]-C[0])*(B[1]-A[1])
    ua = (D[0]-C[0])*(A[1]-C[1])-(D[1]-C[1])*(A[0]-C[0])
    ub = (B[0]-A[0])*(A[1]-C[1])-(B[1]-A[1])*(A[0]-C[0])
    return q!=0 and 0<ua/q and 0<ub/q and 1>ua/q and 1>ub/q

def triangle_intersect(A,B):
    a01 = intersect(A[0],A[1],B[0],B[1]) or intersect(A[0],A[1],B[0],B[2]) or intersect(A[0],A[1],B[1],B[2])
    a02 = intersect(A[0],A[2],B[0],B[1]) or intersect(A[0],A[2],B[0],B[2]) or intersect(A[0],A[2],B[1],B[2])
    a12 = intersect(A[1],A[2],B[0],B[1]) or intersect(A[1],A[2],B[0],B[2]) or intersect(A[1],A[2],B[1],B[2])
    return a01 or a02 or a12

def same_side(p1,p2,a,b):
    cp1 = np.cross(b-a,p1-a)
    cp2 = np.cross(b-a,p2-a)
    return cp1*cp2>=0

def sign(p1,p2,p3):
    return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])

def point_in_triangle(p,t):
    #return same_side(p,t[0],t[1],t[2]) and same_side(p,t[1],t[0],t[2]) and same_side(p,t[2],t[0],t[1])
    if np.array_equal(p,t[0]) or np.array_equal(p,t[1]) or np.array_equal(p,t[2]):
        return False, True
    b1 = sign(p,t[0],t[1])
    b2 = sign(p,t[1],t[2])
    b3 = sign(p,t[2],t[0])
    return ((b1<0 and b2<0) or (b1>0 and b2>0)) and ((b3<0 and b2<0) or (b3>0 and b2>0)), ((b1<=0 and b2<=0) or (b1>=0 and b2>=0)) and ((b3<=0 and b2<=0) or (b3>=0 and b2>=0))

A = np.array([[0,0],[0,1],[1,1]])
B = np.array([[0,1],[1,0],[1,1]])

#print(point_in_triangle([0.5,0],A))

w = 6
h = 2
n = w*h
tris = 0
p = [math.floor(w/2),math.floor(h/2)]

for a in xrange(n):
    A = np.array(divmod(a,h))
    for b in xrange(n):
        if a!=b:
            B = np.array(divmod(b,h))
            for c in xrange(n):
                if a!=c and b!=c:
                    C = np.array(divmod(c,h))
                    if point_in_triangle(p,[A,B,C])[1]:
                        tris+=1
                        
print(tris)