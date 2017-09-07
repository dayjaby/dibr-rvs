import numpy as np
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt
import os
import OpenEXR
import Imath
import time
import math
import array
import util

import theano.tensor as T
import theano
from theano.tensor.nlinalg import matrix_inverse

from optimal_tree import rvs

depthPath = 'depth'
imgPath = 'img'

with open('cameraSettings.json') as file:
    camera_settings = json.load(file)

class Camera:
    def __init__(self,id,settings):
        self.id = id
        self.settings = settings
        self.x = settings['x']
        self.y = settings['y']
        self.P = np.matrix(settings['projection'])
        self.K = np.matrix(settings['kalibration'])
        self.R = np.matrix(settings['rotation'])
        self.T = np.array(settings['translation'])
        self.KR = np.dot(self.K, self.R)
        self.KRinv = np.linalg.inv(self.KR)
        self.KRT = np.dot(self.KR,self.T)
        self.filename = settings['file']
        self.colorImg = Image.open(os.path.join(imgPath,self.filename+".png"))
        self.depthImgEXR = OpenEXR.InputFile(os.path.join(depthPath,self.filename+".exr"))
        dw = self.dataWindow = self.depthImgEXR.header()['dataWindow']
        self.width, self.height = self.size = (dw.max.x-dw.min.x+1,dw.max.y-dw.min.y+1)
        self.depthImg = Image.frombytes("F",self.size,self.depthImgEXR.channel('R',Imath.PixelType(Imath.PixelType.FLOAT)))
        self.depthPixel = np.array(self.depthImg.getdata()).reshape((self.height,self.width))
        self.colorPixel = np.array(self.colorImg.getdata()).reshape((self.height,self.width,4))

    def __eq__(self,other):
        return self.filename == other.filename

    def render(self,filename):
        self.depthImg.save(filename)

    def cvtGray(self):
        if not hasattr(self,'gray'):
            self.gray = cv2.cvtColor(np.array(self.colorImg.convert('RGB')),cv2.COLOR_RGB2GRAY)
        return self.gray

    def calcOpticalFlowFarneback(self,c):
        if not hasattr(self,'opticalFlowFarneback'):
            self.opticalFlowFarneback = {}
        if c.id not in self.opticalFlowFarneback:
            self.opticalFlowFarneback[c.id] = cv2.calcOpticalFlowFarneback(self.cvtGray(),c.cvtGray(), 0.5, 3, 15, 3, 5, 1.1, 0)
        return self.opticalFlowFarneback[c.id]


def fill(img,x,y,z,width,height,mode='fillWhite'):
    img[math.floor(x),math.floor(y)] = z

class DIBR:
    adjust = T.scalar('adjust')
    KR = T.matrix('KR')
    KRinv = T.matrix('KRinv')
    KRT = T.matrix('KRT')
    KR2 = T.matrix('KR2')
    KRT2 = T.matrix('KRT2')
    pix = T.matrix('pix')
    xys = T.imatrix('xys')
    srcColor = T.tensor3('srcColor')
    coords = theano.shared(np.ndarray(shape=(0,0)))
    coordsShape = (0,0)

    @staticmethod
    def _imageWarpPixel(xy,KR,KRT,KR2,KRT2,pix):
        x,y = xy[0], xy[1]
        xy1 = T.as_tensor_variable([x,y,1])
        v = pix[xy1[1],xy1[0]] * xy1
        p = T.dot(matrix_inverse(KR), T.transpose(v+KRT))
        return (T.transpose(T.dot(KR2,p)) - KRT2)[0]

    _imageWarpParams = [KR,KRT,KR2,KRT2,pix]

    _imageWarpScanResult, _imageWarpScanUpdates = theano.scan(
                    fn=_imageWarpPixel.__func__,
                    outputs_info=None,
                    sequences=[xys],
                    non_sequences=_imageWarpParams)
    _imageWarpGPU = theano.function(inputs=_imageWarpParams+[xys],outputs=_imageWarpScanResult)

    pixT = T.transpose(pix)
    dest_img = T.zeros_like(pixT)
    pix2 = pixT.flatten()[:,np.newaxis]
    # warp from camera 1 to camera 2
    r2 = T.dot(pix2 * coords + KRT[0],T.dot(KRinv,KR2)) - KRT2[0]

    # only take the points with 0<=x<pix.shape[1] and 0<=y<pix.shape[0]
    xge0 = (r2[:,0]>=0).nonzero()
    r2 = r2[xge0]
    c = coords[xge0]
    yge0 = (r2[:,1]>=0).nonzero()
    r2 = r2[yge0]
    c = c[yge0]
    xltw = (r2[:,0]/r2[:,2]<T.shape(pix)[1]-adjust).nonzero()
    r2 = r2[xltw]
    c = c[xltw]
    ylth = (r2[:,1]/r2[:,2]<T.shape(pix)[0]-adjust).nonzero()
    r2 = r2[ylth]
    c = c[ylth]

    resultOrigAndWarp = T.concatenate((c[:,0:2],r2),axis=1)
    _imageWarp2GPU = theano.function([pix,KRT,KRinv,KR2,KRT2,adjust],resultOrigAndWarp,on_unused_input='ignore')
    #theano.printing.debugprint(_imageWarp2GPU)

    rxf = r2[:,0]/r2[:,2]
    ryf = r2[:,1]/r2[:,2]
    rx = T.cast(rxf,'int32')
    ry = T.cast(ryf,'int32')
    dest_img = T.set_subtensor(dest_img[rx,ry],r2[:,2])
    _imageWarp2GPUFilled = theano.function([pix,KRT,KRinv,KR2,KRT2,adjust],T.transpose(dest_img))

    # interpolate each pixel with the color values at (x,y),(x,y+1),(x+1,y),(x+1,y+1)
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

    @staticmethod
    def _assureProperCoords(c):
        if c.width != DIBR.coordsShape[0] or c.height != DIBR.coordsShape[1]:
            newCoords = np.array(np.meshgrid(xrange(0,c.width),xrange(0,c.height),[1]),dtype=np.int32).T.reshape(-1,3)
            DIBR.coords.set_value(newCoords)
            DIBR.coordsShape = (c.width,c.height)

    @staticmethod
    def _imageWarpFilled(c1,c2,pix):
        return DIBR._imageWarp2GPUFilled(pix,c1.KRT,c1.KRinv,c2.KR,c2.KRT,0)

    @staticmethod
    def _imageWarpColor(c1,c2,depth,color):
        return DIBR._imageWarp2GPUColor(depth,color,c1.KRT,c1.KRinv,c2.KR,c2.KRT,1)

    @staticmethod
    def InverseMapping(src,dest):
        start = time.time()
        src = src[0]
        DIBR._assureProperCoords(src)
        depthWarpedData = DIBR._imageWarpFilled(src,dest,src.depthPixel)
        kernel5 = np.ones((5,5),np.uint8)
        kernel3 = np.ones((3,3),np.uint8)
        depthWarpedData = cv2.dilate(depthWarpedData, kernel5, iterations=1)
        depthWarpedData = cv2.erode(depthWarpedData, kernel3, iterations=2)
        depthWarped = Image.fromarray(depthWarpedData.reshape(c1.height,c1.width))
        imgWarpedData = DIBR._imageWarpColor(dest,src,depthWarpedData,src.colorPixel)
        imgWarped = Image.fromarray(np.uint8(imgWarpedData.reshape(c1.height,c1.width,4)),'RGBA')
        end = time.time()
        print("DIBR.InverseMapping in {}s".format(end-start))
        return imgWarped, depthWarped

    @staticmethod
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
        coordinates_vec = DIBR._imageWarpGPU(c1.KR,c1.KRT,c2.KR,c2.KRT,pix,coords)
        result = np.concatenate((coords,coordinates_vec),axis=1)
        """
        # Even better GPU solution (~0.08s per image pair)
        DIBR._assureProperCoords(c1)
        return DIBR._imageWarp2GPU(pix,c1.KRT,c1.KRinv,c2.KR,c2.KRT,1)

    @staticmethod
    def ImageWarp(src,dest):
        start = time.time()
        src = src[0]
        depthWarped = Image.new("F",src.size,"black")
        depthWarpedData = depthWarped.load()
        imgWarped = Image.new("RGB",c1.size,"black")
        imgWarpedData = imgWarped.load()
        result = DIBR._imageWarp(src,dest,src.depthPixel)
        for ox,oy,x,y,z in result:
            fill(depthWarpedData,x/z,y/z,z,src.width,src.height,mode='forward')
            for xf in [math.floor,math.ceil]:
                for yf in [math.floor,math.ceil]:
                    if xf(x/z)>=0 and xf(x/z)<src.width and yf(y/z)>=0 and yf(y/z)<src.height:
                        imgWarpedData[xf(x/z),yf(y/z)] =tuple( src.colorPixel[oy,ox].astype(int))
        end = time.time()
        print("DIBR.ImageWarp in {}ms".format(end-start))
        return imgWarped, depthWarped

class DIBRCamera(Camera):
    def __init__(self,id,settings):
        Camera.__init__(self,id,settings)
        self.referenceViews = []
        #self.DIBR_method = DIBR.ImageWarp
        self.DIBR_method = DIBR.InverseMapping

    def addReference(self,cam):
        self.referenceViews.append(cam)

    def setReference(self,cam):
        self.referenceViews = [cam]

    def render(self,imgFilename,filename):
        img, depth  = self.DIBR_method(self.referenceViews,self)
        if depth is not None:
            util.exportGrayEXR(filename,depth.getdata(), depth.size)
        if img is not None:
            img.save(imgFilename)

cameras = {id: Camera(id,settings) for id,settings in enumerate(camera_settings)}
xs = list({c.x for id,c in cameras.items()})
ys = list({c.y for id,c in cameras.items()})
xyCamera = {(c.x,c.y) : c for id,c in cameras.items()}

def w_image_flow_2_row(y):
    def w_image_flow_2(v,r1,r2):
        cs = [xyCamera[(xs[x],y)] for x in [r1,v,r2]]
        flows = [cs[1].calcOpticalFlowFarneback(cs[i]) for i in [0,2]]
        norms = np.array([np.linalg.norm(flow.reshape(-1,2),axis=1) for flow in flows])
        x = np.mean(np.minimum(norms[0],norms[1]))
        err = x/(x+1)
        return err
    return w_image_flow_2


def w_image_flow_1_row(y):
    def w_image_flow_1(v,r,_):
        c1 = xyCamera[(xs[r],y)]
        c2 = xyCamera[(xs[v],y)]
        prev = c1.cvtGray()
        next = c2.cvtGray()
        flow = cv2.calcOpticalFlowFarneback(prev,next, 0.5, 3, 15, 3, 5, 1.1, 0)
        x = (np.mean(np.linalg.norm(flow.reshape(-1,2),axis=1)))
        err = x/(x+1)
        return err
    return w_image_flow_1

for c2id, c2 in cameras.items():
    dibrCam = DIBRCamera(c2.id,c2.settings)
    for c1id, c1 in cameras.items():
        dibrCam.setReference(c1)
        if c1!=c2:
            if abs(c1.x - c2.x)<2 and abs(c1.y - c2.y) < 2:
                dibrCam.render("dibr-simple-results/dibr_{}_{}.png".format(c1.id,c2.id),"dibr-simple-results/intersection_{}_{}.exr".format(c1.id,c2.id))
                break
    break

"""m = 4
print("n={}, m={}".format(len(xs),m))
rvs_dijkstra, worst, best = rvs(len(xs),m=m,k=2)
print("#gammas worst:{}, best:{}".format(worst,best))
for y in ys:
    best_rvs, avg = rvs_dijkstra(weight_fn=w_image_flow_2_row(y))
    print(best_rvs)
    print("#gammas:{}".format(avg))
    best_rvs, avg = rvs_dijkstra(weight_fn=None)
    print(best_rvs)
    print("#gammas:{}".format(avg))
"""
