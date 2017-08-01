import numpy as np
import json
from PIL import Image
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
        self.depthImg = Image.fromstring("F",self.size,self.depthImgEXR.channel('R',Imath.PixelType(Imath.PixelType.FLOAT)))
        self.depthPixel = self.depthImg.load()
        self.colorPixel = np.array(self.colorImg.getdata()).reshape((self.height,self.width,4))

    def __eq__(self,other):
        return self.filename == other.filename

    def render(self,filename):
        self.depthImg.save(filename)

def fill(img,x,y,z,width,height,mode='fillWhite'):
    for xf in [math.floor,math.ceil]:
        for yf in [math.floor,math.ceil]:
            wx = xf(x)
            wy = yf(y)
            if wx>=0 and wx<width and wy>=0 and wy<height:
                if mode == 'fillWhite':
                    img[wx,wy] = 1
                elif mode == 'forward':
                    if img[wx,wy] == 0 or img[wx,wy]>z:
                        img[wx,wy] = z




class DIBR:
    KR = T.matrix('KR')
    KRT = T.matrix('KRT')
    KR2 = T.matrix('KR2')
    KRT2 = T.matrix('KRT2')
    pix = T.matrix('pix')
    xys = T.imatrix('xys')

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
        coords = np.array(np.meshgrid(xrange(0,c1.width),xrange(0,c1.height)),dtype=np.int32).T.reshape(-1,2)
        coordinates_vec = DIBR._imageWarpGPU(c1.KR,c1.KRT,c2.KR,c2.KRT,pix,coords)
        return np.concatenate((coords,coordinates_vec),axis=1)

    @staticmethod
    def ImageWarp(src,dest):
        start = time.time()
        depthWarped = Image.new("F",src.size,"black")
        depthWarpedData = depthWarped.load()
        imgWarped = Image.new("RGB",c1.size,"black")
        imgWarpedData = imgWarped.load()
        pix = np.array(src.depthImg.getdata()).reshape((src.size[1],src.size[0]))
        result = DIBR._imageWarp(src,dest,pix)
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
        depthWarped = Image.new("F",c1.size,"black")
        depthPix = np.array(src.depthImg.getdata()).reshape((src.size[1],src.size[0]))
        depthWarpedData = depthWarped.load()
        for ox,oy,x,y,z in DIBR._imageWarp(src,dest,depthPix):
            fill(depthWarpedData,x/z,y/z,z,src.width,src.height,mode='forward')
        imgWarped = Image.new("RGB",c1.size,"black")
        depthPix2 = np.array(depthWarped.getdata()).reshape((src.size[1],src.size[0]))
        imgWarpedData = imgWarped.load()
        for ox,oy,x,y,z in DIBR._imageWarp(dest,src,depthPix2):
            xmin = math.floor(x/z)
            xmax = xmin+1
            ymin = math.floor(y/z)
            ymax = ymin+1
            if xmin>=0 and ymin>=0 and xmax<dest.width and ymax<dest.height:
                v1 = (x/z-xmin) * src.colorPixel[ymax,xmax] + (xmax-x/z) * src.colorPixel[ymax,xmin]
                v2 = (x/z-xmin) * src.colorPixel[ymin,xmax] + (xmax-x/z) * src.colorPixel[ymin,xmin]
                v = np.round((y/z-ymin) * v1 + (ymax-y/z) * v2).astype(int)
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
        if len(self.referenceViews) == 1:
            img, depth  = self.DIBR_method(self.referenceViews[0],self)
            util.exportGrayEXR(filename,depth.getdata(), depth.size)
            img.save(imgFilename)

cameras = [Camera(id,settings) for id,settings in enumerate(camera_settings)]
for c2 in cameras:
    dibrCam = DIBRCamera(c2.id,c2.settings)
    for c1 in cameras:
        dibrCam.setReference(c1)
        if c1!=c2 and abs(c1.x - c2.x)<2 and abs(c1.y - c2.y)<2:
            dibrCam.render("dibr-simple-results/dibr_{}_{}.png".format(c1.id,c2.id),"dibr-simple-results/intersection_{}_{}.png".format(c1.id,c2.id))
            break
    break
