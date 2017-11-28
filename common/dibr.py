import theano.tensor as T
import theano
from theano.compile.ops import as_op
from theano.tensor.nlinalg import matrix_inverse
theano.config.warn.round=False
#theano.config.mode='FAST_COMPILE'
import numpy as np
import cv2
import json
from PIL import Image
import time
import math

adjust = T.scalar('adjust')
KR = T.matrix('KR')
KRinv = T.matrix('KRinv')
KRT = T.matrix('KRT')
KR2 = T.matrix('KR2')
KRT2 = T.matrix('KRT2')
pix = T.matrix('pix')
d1 = T.matrix('d1')
d2 = T.matrix('d2')
i1 = T.tensor3('i1')
i2 = T.tensor3('i2')
xys = T.imatrix('xys')
srcColor = T.tensor3('srcColor')
coords = theano.shared(np.ndarray(shape=(0,0)))
coordsShape = (0,0)

pixT = T.transpose(pix)
dest_img = T.zeros_like(pixT)
pix2 = pixT.flatten()[:,np.newaxis]
# warp from camera 1 to camera 2
r2 = T.dot(pix2 * coords + KRT[0],T.dot(KRinv,KR2)) - KRT2[0]


# only take the points with 0<=x<pix.shape[1] and 0<=y<pix.shape[0]
xge0 = T.and_(T.and_(r2[:,0]/r2[:,2]<T.shape(pix)[1]-adjust,r2[:,1]/r2[:,2]<T.shape(pix)[0]-adjust),T.and_(r2[:,0]>=0,r2[:,1]>=0)).nonzero()
#xge0 = (r2[:,0]>=0).nonzero()
r2 = r2[xge0]
c = coords[xge0]
p = pix2[xge0]
"""yge0 = (r2[:,1]>=0).nonzero()
r2 = r2[yge0]
c = c[yge0]
xltw = (r2[:,0]/r2[:,2]<T.shape(pix)[1]-adjust).nonzero()
r2 = r2[xltw]
c = c[xltw]
ylth = (r2[:,1]/r2[:,2]<T.shape(pix)[0]-adjust).nonzero()
r2 = r2[ylth]
c = c[ylth]"""
rx = r2[:,0]/r2[:,2]
ry = r2[:,1]/r2[:,2]
_imageWarp2GPUFilled2 = theano.function([pix,KRT,KRinv,KR2,KRT2,adjust],T.concatenate((c[:,0:2],rx[:,np.newaxis],ry[:,np.newaxis],r2[:,2,np.newaxis]),axis=1))

resultOrigAndWarp = T.set_subtensor(r2[(r2<0.0001).nonzero()],999999)
resultOrigAndWarp = T.concatenate((c[:,0:2],resultOrigAndWarp),axis=1)
_imageWarp2GPU = theano.function([pix,KRT,KRinv,KR2,KRT2,adjust],resultOrigAndWarp,on_unused_input='ignore')

#theano.printing.debugprint(_imageWarp2GPU)


r3 = r2
rd = r3[:,2]
rxf = r3[:,0]/rd
ryf = r3[:,1]/rd
rx = T.cast(rxf,'int32')
ry = T.cast(ryf,'int32')
#rxy = T.transpose(T.as_tensor_variable([rx,ry]))
i = T.argsort(-rd)
rx = rx[i]
ry = ry[i]
rd = rd[i]
c = c[i]
p = p[i]
r2 = r2[i]
dest_img2 = T.set_subtensor(dest_img[rx,ry],p[:,0])
dest_img = T.set_subtensor(dest_img[rx,ry],rd)
_imageWarp2GPUFilled2 = theano.function([pix,KRT,KRinv,KR2,KRT2,adjust],T.transpose(dest_img2))

_imageWarp2GPUFilled = theano.function([pix,KRT,KRinv,KR2,KRT2,adjust],T.transpose(dest_img))

# interpolate each pixel with the color values at (x,y),(x,y+1),(x+1,y),(x+1,y+1)
rxf = r2[:,0]/r2[:,2]
ryf = r2[:,1]/r2[:,2]
rx = T.cast(rxf,'int32')
ry = T.cast(ryf,'int32')
rxp1 = rx+1
ryp1 = ry+1
color0 = srcColor[ry  ,rx  ]
color1 = srcColor[ryp1,rx  ]
color2 = srcColor[ry  ,rxp1]
color3 = srcColor[ryp1,rxp1]
rxfmrx = rxf-rx
rxp1mrxf = rxp1-rxf
y1 = rxfmrx * T.transpose(color3) + rxp1mrxf * T.transpose(color1)
y0 = rxfmrx * T.transpose(color2) + rxp1mrxf * T.transpose(color0)
shape = T.as_tensor_variable([T.shape(pix)[1],T.shape(pix)[0],4])
v = T.transpose(T.cast((ryf-ry)*y1+(ryp1-ryf)*y0,'int32'))
dest_img = T.zeros(shape,'int32')
oc = T.cast(c[:,0:2],'int32')
ocx = oc[:,0]
ocy = oc[:,1]
dest_img = T.transpose(T.set_subtensor(dest_img[ocx,ocy],v),(1,0,2))
_imageWarp2GPUColor = theano.function([pix,srcColor,KRT,KRinv,KR2,KRT2,adjust],dest_img,on_unused_input='ignore')

i1r = i1[:,:,0].flatten()
i1g = i1[:,:,1].flatten()
i1b = i1[:,:,2].flatten()
d1f = d1.flatten()
d1fzero = (d1f<0.0001).nonzero()
d1f = T.set_subtensor(d1f[d1fzero],999999)
i2r = i2[:,:,0].flatten()
i2g = i2[:,:,1].flatten()
i2b = i2[:,:,2].flatten()
d2f = d2.flatten()
d1d2ltepsilon_cond = T.and_(T.abs_(d1f-d2f)<0.05,T.and_(d2f>0,d1f>0))
d1d2ltepsilon = d1d2ltepsilon_cond.nonzero()
d1f = T.set_subtensor(d1f[d1d2ltepsilon],(d1f[d1d2ltepsilon]+d2f[d1d2ltepsilon])/2)
i1r = T.set_subtensor(i1r[d1d2ltepsilon],(i1r[d1d2ltepsilon]+i2r[d1d2ltepsilon])/2)
i1g = T.set_subtensor(i1g[d1d2ltepsilon],(i1g[d1d2ltepsilon]+i2g[d1d2ltepsilon])/2)
i1b = T.set_subtensor(i1b[d1d2ltepsilon],(i1b[d1d2ltepsilon]+i2b[d1d2ltepsilon])/2)
d2ltd1 = T.and_(d2f<d1f,T.and_(d2f>0,T.invert(d1d2ltepsilon_cond))).nonzero()
d1f = T.set_subtensor(d1f[d2ltd1],d2f[d2ltd1])
i1r = T.set_subtensor(i1r[d2ltd1],i2r[d2ltd1])
i1g = T.set_subtensor(i1g[d2ltd1],i2g[d2ltd1])
i1b = T.set_subtensor(i1b[d2ltd1],i2b[d2ltd1])
i1f = T.as_tensor_variable([i1r,i1g,i1b,i1[:,:,3].flatten()])
i1f = T.set_subtensor(i1f[3,:],255)
_combineTwoImages = theano.function([i1,d1,i2,d2],(T.transpose(i1f),d1f),on_unused_input='ignore')

i1r = i1[:,:,0].flatten()
i1g = i1[:,:,1].flatten()
i1b = i1[:,:,2].flatten()
i2r = i2[:,:,0].flatten()
i2g = i2[:,:,1].flatten()
i2b = i2[:,:,2].flatten()
blackPixel = ((i1r+i1g+i1b)<1).nonzero()
nonblackPixel = ((i1r+i1g+i1b)>0).nonzero()
i1r = T.set_subtensor(i1r[blackPixel],i2r[blackPixel])
i1g = T.set_subtensor(i1g[blackPixel],i2g[blackPixel])
i1b = T.set_subtensor(i1b[blackPixel],i2b[blackPixel])
i1r = T.set_subtensor(i1r[nonblackPixel],(i1r[nonblackPixel]+i2r[nonblackPixel])/2)
i1g = T.set_subtensor(i1g[nonblackPixel],(i1g[nonblackPixel]+i2g[nonblackPixel])/2)
i1b = T.set_subtensor(i1b[nonblackPixel],(i1b[nonblackPixel]+i2b[nonblackPixel])/2)
i1f = T.as_tensor_variable([i1r,i1g,i1b,i1[:,:,3].flatten()])
i1f = T.set_subtensor(i1f[3,:],255)
_combineTwoImagesColor = theano.function([i1,i2],T.transpose(i1f),on_unused_input='ignore')

def _assureProperCoords(c):
    global coords, coordsShape
    if c.width != coordsShape[0] or c.height != coordsShape[1]:
        newCoords = np.array(np.meshgrid(xrange(0,c.width),xrange(0,c.height),[1]),dtype=np.int32).T.reshape(-1,3)
        coords.set_value(newCoords)
        coordsShape = (c.width,c.height)

def _imageWarpFilled(c1,c2,pix):
    return _imageWarp2GPUFilled(pix,c1.KdRT,c1.KdRinv,c2.KdR,c2.KdRT,1)

def _imageWarpColor(c1,c2,depth,color):
    return _imageWarp2GPUColor(depth,color,c1.KRT,c1.KRinv,c2.KR,c2.KRT,1)

def InverseMapping(src,dest,makeimg=True):
    if isinstance(src,list):
        src = src[0]
    _assureProperCoords(src)
    depthWarpedData = _imageWarpFilled(src,dest,src.depthPixel)
    kernel5 = np.ones((5,5),np.uint8)
    kernel3 = np.ones((3,3),np.uint8)
    #depthWarpedData = cv2.dilate(depthWarpedData, kernel5, iterations=1)
    #depthWarpedData = cv2.erode(depthWarpedData, kernel3, iterations=2)
    depthWarpedData = cv2.morphologyEx(depthWarpedData, cv2.MORPH_CLOSE, kernel3)
    imgWarpedData = _imageWarpColor(dest,src,depthWarpedData,src.colorPixel)
    if makeimg:
        depthWarped = Image.fromarray(depthWarpedData.reshape(src.height,src.width))
        imgWarped = Image.fromarray(np.uint8(imgWarpedData.reshape(src.height,src.width,4)),'RGBA')
        return imgWarped, depthWarped
    else:
        return imgWarpedData, depthWarpedData

def InverseMapping2(srcs,dest):
    src1 = srcs[0]
    src2 = srcs[1]
    i1, d1 = InverseMapping(src1,dest,makeimg=False)
    i2, d2 = InverseMapping(src2,dest,makeimg=False)
    iData, dData  = _combineTwoImages(i1,d1,i2,d2)
    i = Image.fromarray(np.uint8(iData.reshape(src1.height,src1.width,4)),'RGBA')
    d = Image.fromarray(dData.reshape(src1.height,src1.width))
    return i, d

def InverseMappingX(srcs,dest):
    x = [InverseMapping(src,dest,makeimg=False) for src in srcs]
    def combine2(a,n):
        if n>1:
            combine2(a,n/2)
            combine2(a+n,n/2)
        i1, d1 = x[a]
        if a+n < len(x):
            i2, d2 = x[a+n]
            iData, dData  = _combineTwoImages(i1,d1,i2,d2)
            i = Image.fromarray(np.uint8(iData.reshape(srcs[0].height,srcs[0].width,4)),'RGBA')
            d = Image.fromarray(dData.reshape(srcs[0].height,srcs[0].width))
            x[a] = i,d
        
    combine2(0,int(np.power(np.ceil(np.log2(len(srcs))),2)/2))
    return x[0]
    
    
def InverseMappingSimplified(src,dest):
    _assureProperCoords(src)
    imgWarpedData = _imageWarpColor(dest,src,dest.depthPixel,src.colorPixel)
    return imgWarpedData

def InverseMapping2Simplified(src1,src2,dest):
    i1 = InverseMappingSimplified(src1,dest)
    i2 = InverseMappingSimplified(src2,dest)
    i = Image.fromarray(np.uint8(i2.reshape(src1.height,src1.width,4)),'RGBA')
    return i, None
    iData  = _combineTwoImagesColor(i1,i2)
    i = Image.fromarray(np.uint8(iData.reshape(src1.height,src1.width,4)),'RGBA')
    return i, None

def _imageWarp(c1,c2,pix):
    #temp = np.zeros((c1.size[0],c1.size[1]),dtype=np.float32)
    # Naive solution: using CPU (~14s per image pair)
    """for x in xrange(0,c1.width):
        for y in xrange(0,c1.height):
            position = np.dot(c1.KRinv,(c1.pixel[x,y]*np.array([[x,y,1]])+c1.KRT).transpose()).A1
            coordinates = (np.dot(c2.KR,position)-c2.KRT).transpose().A1
            fill(temp,coordinates[0]/coordinates[2],coordinates[1]/coordinates[2],c1.width,c1.height)"""
    # Better solution: using GPU (~5s per image pair)
    """
    coords = np.array(np.meshgrid(xrange(0,c1.width),xrange(0,c1.height)),dtype=np.int32).T.reshape(-1,2)
    coordinates_vec = _imageWarpGPU(c1.KR,c1.KRT,c2.KR,c2.KRT,pix,coords)
    result = np.concatenate((coords,coordinates_vec),axis=1)
    """
    # Even better GPU solution (~0.08s per image pair)
    _assureProperCoords(c1)
    return _imageWarp2GPU(pix,c1.KRT,c1.KRinv,c2.KR,c2.KRT,1)



def fill(img,x,y,z,width,height,mode='fillWhite'):
    img[math.floor(x),math.floor(y)] = z


def ImageWarp(src,dest):
    start = time.time()
    src = src[0]
    depthWarped = Image.new("F",src.size,"black")
    depthWarpedData = depthWarped.load()
    imgWarped = Image.new("RGB",c1.size,"black")
    imgWarpedData = imgWarped.load()
    result = _imageWarp(src,dest,src.depthPixel)
    for ox,oy,x,y,z in result:
        fill(depthWarpedData,x/z,y/z,z,src.width,src.height,mode='forward')
        for xf in [math.floor,math.ceil]:
            for yf in [math.floor,math.ceil]:
                if xf(x/z)>=0 and xf(x/z)<src.width and yf(y/z)>=0 and yf(y/z)<src.height:
                    imgWarpedData[xf(x/z),yf(y/z)] =tuple( src.colorPixel[oy,ox].astype(int))
    end = time.time()
    return imgWarped, depthWarped