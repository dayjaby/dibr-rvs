import numpy as np
import cv2
import json
from PIL import Image
import os
import OpenEXR
import Imath
from numpy import genfromtxt
from . import dibr, util

class Camera:
    def __init__(self,id,settings):
        self.id = id
        self.settings = settings
        self.x = settings['x']
        self.y = settings['y']
        #self.P = np.matrix(settings['projection'])
        self.K = np.matrix(settings['kalibration'])
        self.Kd = np.matrix(settings['kalibration-depth']) if 'kalibration-depth' in settings else self.K
        self.R = np.matrix(settings['rotation'])
        self.T = np.array(settings['translation'])
        self.KR = np.dot(self.K, self.R)
        self.KRinv = np.linalg.inv(self.KR)
        self.KRT = np.dot(self.KR,self.T)
        self.KdR = np.dot(self.Kd, self.R)
        self.KdRinv = np.linalg.inv(self.KdR)
        self.KdRT = np.dot(self.KdR,self.T)
        self.img_file = settings['img_file']
        self.img_directory = settings['img_directory']
        self.depth_file = settings['depth_file']
        self.depth_directory = settings['depth_directory']
        self.colorImg = Image.open(os.path.join(self.img_directory,self.img_file))
        if self.depth_file.endswith('.mat'):
            self.width, self.height = self.size = self.colorImg.size
            self.depthPixel = genfromtxt(os.path.join(self.depth_directory,self.depth_file),delimiter=' ')
            self.dheight, self.dwidth = self.depthPixel.shape
        else:
            self.depthImgEXR = OpenEXR.InputFile(os.path.join(self.depth_directory,self.depth_file))
            dw = self.dataWindow = self.depthImgEXR.header()['dataWindow']
            self.dwidth, self.dheight = self.width, self.height = self.size = (dw.max.x-dw.min.x+1,dw.max.y-dw.min.y+1)
            self.depthImg = Image.frombytes("F",self.size,self.depthImgEXR.channel('R',Imath.PixelType(Imath.PixelType.FLOAT)))
            self.depthPixel = np.array(self.depthImg.getdata(),dtype=np.float32).reshape((self.height,self.width))
        self.colorPixel = np.array(self.colorImg.getdata(),dtype=np.uint8)
        if np.prod(self.colorPixel.shape) == self.height*self.width*4:
            self.colorPixel = self.colorPixel.reshape((self.height,self.width,4))
        else:
            rshp = self.colorPixel.reshape((self.height,self.width,3))
            self.colorPixel = np.full((self.height,self.width,4),255,dtype=np.uint8)
            self.colorPixel[:,:,0:3]=rshp

        if "distortion" in settings and settings["distortion"] is not None:
            self.colorPixel = cv2.undistort(self.colorPixel,self.K,np.array(settings["distortion"]))
        if "distortion-depth" in settings and settings["distortion-depth"] is not None:
            factor = 2.54716981
            cp = cv2.undistort(self.depthPixel,self.Kd,np.array(settings["distortion-depth"]))
            self.depthPixel = np.full((self.height,self.width),0,dtype=np.float32)
            self.depthPixel[0:self.dheight,0:self.dwidth] = cp
            for x in xrange(20):
                r = np.dot(np.array([self.depthPixel[100,x]/1000.0*np.array([x,100,1])]) + self.KdRT[0],np.dot(self.KdRinv,self.KR)) - self.KRT[0]
                print(r)
            dibr._assureProperCoords(self)
            self.depthPixel = dibr._imageWarp2GPUFilled(self.depthPixel/1000.0,self.KdRT,self.KdRinv,self.KR,self.KRT,1)
            #for i in xrange(270):
            #    cx, cy, rx, ry, rd = self.depthPixel[i]
            #    print("{},{} -> {},{}".format(cx,cy,rx/rd,ry/rd))
                
            cv2.imshow('d',self.depthPixel)
            cv2.waitKey(0)

        
        #print(self.colorPixel.shape,self.depthPixel.shape)
        #print(self.colorPixel.dtype,self.depthPixel.dtype)

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
    
class DIBRCamera(Camera):
    def __init__(self,id,settings):
        Camera.__init__(self,id,settings)
        self.referenceViews = []
        self.DIBR_method = dibr.InverseMapping

    def addReference(self,cam):
        self.referenceViews.append(cam)
        if len(self.referenceViews)==2:
            self.DIBR_method = dibr.InverseMapping2
        elif len(self.referenceViews)==1:
            self.DIBR_method = dibr.InverseMapping
        else:
            self.DIBR_method = dibr.InverseMappingX

    def setReferences(self,cams):
        self.referenceViews = cams
        if len(self.referenceViews)==2:
            self.DIBR_method = dibr.InverseMapping2
        elif len(self.referenceViews)==1:
            self.DIBR_method = dibr.InverseMapping
        else:
            self.DIBR_method = dibr.InverseMappingX

    def render(self,imgFilename,filename=""):
        self.img, self.depth  = self.DIBR_method(self.referenceViews,dest=self)
        if self.depth is not None:
            util.exportGrayEXR(filename,self.depth.getdata(), self.depth.size)
        if self.img is not None:
            self.img.save(imgFilename)