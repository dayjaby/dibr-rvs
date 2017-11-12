import numpy as np

import pycosat
import time
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

w = 4
h = 4
m = 9
t = time.time()
n = w*h
triangles = dict()
for a in xrange(n):
    A = np.array(divmod(a,h))
    for b in xrange(n):
        if a!=b:
            B = np.array(divmod(b,h))
            for c in xrange(n):
                if a!=c and b!=c:
                    C = np.array(divmod(c,h))
                    x,y,z = sorted([A,B,C],key = lambda x: (x[0],x[1]))
                    norms = [np.linalg.norm(A-B),np.linalg.norm(A-C),np.linalg.norm(B-C)]
                    if np.max(norms) < 9.1:
                        area = np.linalg.det(np.array([[1,x[0],x[1]],[1,y[0],y[1]],[1,z[0],z[1]]]))
                        #if np.abs(area)>=3:
                        triangles[(x[0],x[1],y[0],y[1],z[0],z[1])] = np.array([x,y,z])

T = triangles.values()
lT = len(T)
data = {}
cnf = []
print(time.time()-t)
t = time.time()

segments = dict()

for i,A in enumerate(T):
    for j in xrange(3):
        a,b = sorted([j,(j+1)%3])
        tAj = tuple([tuple(A[a]),tuple(A[b])])
        if tAj not in segments:
            segments[tAj] = []
        segments[tAj].append(i)

for i, s in enumerate(segments.items()):
    seg, tris = s
    for j, seg2 in enumerate(segments.keys()):
        if i<j:
            if intersect(seg[0],seg[1],seg2[0],seg2[1]):
                cnf.append([-n-lT-i-1,-n-lT-j-1])
    clause = [-n-lT-i-1]
    for tri in tris:
        if n+tri+1 not in data:
            data[n+tri+1] = []
        data[n+tri+1].append(i+1)
        clause.append(n+tri+1)
        cnf.append([-n-tri-1,+n+lT+i+1])
    data[n+lT+i+1] = seg
    cnf.append(clause)
    
lS = len(segments)

for c in itertools.combinations(xrange(-n,0),m+1):
    cnf.append(list(c))
print(time.time()-t)
t = time.time()
    
for x,y in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]:
    a = x*h+y
    cnf.append([a+1])
print(time.time()-t)
t = time.time()

for a in xrange(n):
    A = np.array(divmod(a,h))
    clause = []
    clause2 = [-a-1]
    clause3 = []
    for i,tri in enumerate(T):
        x,y = point_in_triangle(A,tri)
        if y:
            clause.append(n+i+1)
        if x:
            cnf.append([-n-i-1,-a-1])
            clause3.append(n+i+1)
        if np.array_equal(tri[0],A) or np.array_equal(tri[1],A) or np.array_equal(tri[2],A):
            clause2.append(n+i+1)
    if len(clause)>0:
        # At least one triangle contains A
        cnf.append(clause)
        # At most one triangle contains A
        for x in clause3:
            for y in clause3:
                if x<y:
                    cnf.append([-x,-y])
    cnf.append(clause2)
print(time.time()-t)
t = time.time()

for i,A in enumerate(T):
    for p in A:
        a = p[0]*h+p[1]
        cnf.append([-n-i-1,a+1])
    """for j,B in enumerate(T):
        if i<j:
            #if triangle_intersect(A,B) or point_in_triangle(A[0],B) or point_in_triangle(B[0],A):
            if point_in_triangle(A[0],B) or point_in_triangle(B[0],A):
                cnf.append([-n-i-1,-n-j-1])"""
                
print(time.time()-t)
t = time.time()
print(len(T))
print(len(cnf))
print([1,n],[n+1,n+lT],[n+lT+1,lS+lT+n])
t = time.time()
tri = set()
i = 0
for sol in pycosat.itersolve(cnf):
    #print("========================")
    i+=1
    s = set()
    for x in sol:
        if x>0:
            if x<n+1:
                #print("Reference view: {}, {}".format(x,np.array(divmod(x-1,h))))
                pass
            elif x<n+lT+1:
                tri.add(x-n-1)
                #print("Triangle: {}, {}".format(x-n,data[x]))
                for p in T[x-n-1]:
                    s.add(tuple(p))
            else:
                #print("Segment: {}, {}".format(x-n-lT,data[x]))
                pass

            
print(len(tri))
print(i)
print(time.time()-t)