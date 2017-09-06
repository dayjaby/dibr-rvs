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
    KR = T.matrix('KR')
    KRinv = T.matrix('KRinv')
    KRT = T.matrix('KRT')
    KR2 = T.matrix('KR2')
    KRT2 = T.matrix('KRT2')
    pix = T.matrix('pix')
    xys = T.imatrix('xys')
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
    r2 = pix2 * coords
    r2 = r2 + KRT[0]
    r2 = T.dot(r2,T.dot(KRinv,KR2))
    r2 = r2 - KRT2[0]
    xge0 = (r2[:,0]>=0).nonzero()
    r2 = r2[xge0]
    c = coords[xge0]
    yge0 = (r2[:,1]>=0).nonzero()
    r2 = r2[yge0]
    c = c[yge0]
    xltw = (r2[:,0]/r2[:,2]<T.shape(pix)[1]).nonzero()
    r2 = r2[xltw]
    c = c[xltw]
    ylth = (r2[:,1]/r2[:,2]<T.shape(pix)[0]).nonzero()
    r2 = r2[ylth]
    c = c[ylth]
    resultOrigAndWarp = T.concatenate((c[:,0:2],r2),axis=1)

    _imageWarp2GPU = theano.function([pix,KRT,KRinv,KR2,KRT2],resultOrigAndWarp,on_unused_input='ignore')
    #theano.printing.debugprint(_imageWarp2GPU)

    rx = T.cast(r2[:,0]/r2[:,2],'int64')
    ry = T.cast(r2[:,1]/r2[:,2],'int64')
    dest_img = T.set_subtensor(dest_img[rx,ry],r2[:,2])
    _imageWarp2GPUFilled = theano.function([pix,KRT,KRinv,KR2,KRT2],T.transpose(dest_img))

    @staticmethod
    def _imageWarp(c1,c2,pix,method=None):
        if method is None:
            method = DIBR._imageWarp2GPU
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
        # Even better GPU solution:
        if c1.width != DIBR.coordsShape[0] or c1.height != DIBR.coordsShape[1]:
            newCoords = np.array(np.meshgrid(xrange(0,c1.width),xrange(0,c1.height),[1]),dtype=np.int32).T.reshape(-1,3)
            DIBR.coords.set_value(newCoords)
            DIBR.coordsShape = (c1.width,c1.height)
        result = method(pix,c1.KRT,c1.KRinv,c2.KR,c2.KRT)
        return result

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

    @staticmethod
    def InverseMapping(src,dest):
        start = time.time()
        src = src[0]
        depthWarpedData = DIBR._imageWarp(src,dest,src.depthPixel,method=DIBR._imageWarp2GPUFilled)
        depthWarped = Image.fromarray(depthWarpedData.reshape(c1.size))
        print("DIBR.InverseMapping#1 in {}ms".format(time.time()-start))
        imgWarped = Image.new("RGB",c1.size,"black")
        imgWarpedData = imgWarped.load()
        epsilon = 0.3
        for ox,oy,x,y,z in DIBR._imageWarp(dest,src,depthWarpedData):
            x = x/z
            y = y/z
            xmin = int(math.floor(x))
            xmax = xmin+1
            ymin = int(math.floor(y))
            ymax = ymin+1
            if xmin>=0 and ymin>=0 and xmax<dest.width and ymax<dest.height:
                colors, depths = zip(*[(src.colorPixel[y_,x_],src.depthPixel[y_,x_]) for y_ in [ymin,ymax] for x_ in [xmin,xmax]])
                y1 = (x-xmin) * colors[3] + (xmax-x) * colors[1]
                y0 = (x-xmin) * colors[2] + (xmax-x) * colors[0]
                v = np.round((y-ymin) * y1 + (ymax-y) * y0).astype(int)
                imgWarpedData[ox,oy] = tuple(v)
        end = time.time()
        print("DIBR.InverseMapping in {}ms".format(end-start))
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
