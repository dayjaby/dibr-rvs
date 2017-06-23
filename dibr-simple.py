import numpy as np
import json
from PIL import Image
import os
import OpenEXR
import Imath
import time
import math

import theano.tensor as T
import theano
from theano.tensor.nlinalg import matrix_inverse

depthPath = 'depth'

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
        self.imgEXR = OpenEXR.InputFile(os.path.join(depthPath,self.filename+".exr"))
        dw = self.dataWindow = self.imgEXR.header()['dataWindow']
        self.width, self.height = self.size = (dw.max.x-dw.min.x+1,dw.max.y-dw.min.y+1)
        self.img = Image.fromstring("F",self.size,self.imgEXR.channel('R',Imath.PixelType(Imath.PixelType.FLOAT)))
        self.pixel = self.img.load()

    def __eq__(self,other):
        return self.filename == other.filename

    def render(self,filename):
        self.img.save(filename)

def fill(img,x,y,width,height):
    for xf in [math.floor,math.ceil]:
        for yf in [math.floor,math.ceil]:
            wx = xf(x)
            wy = yf(y)
            if wx>=0 and wx<width and wy>=0 and wy<height:
                img[wx,wy] = 1

KR = T.matrix('KR')
KRT = T.matrix('KRT')
KR2 = T.matrix('KR2')
KRT2 = T.matrix('KRT2')
t = T.vector('t')
t2 = T.vector('t2')
v = T.matrix('v')
p = T.dot(matrix_inverse(KR),T.transpose(v+KRT))
c = T.transpose(T.dot(KR2,p)) -  KRT2
#theano.scan
f = theano.function([KR,KRT,KR2,KRT2,v],c)
#f = theano.function([KR,KRT,v],p)

class DIBR:
    @staticmethod
    def ImageWarp(c1,c2):
        start = time.time()
        tempImage = Image.new("1",c1.size,"black")
        temp = tempImage.load()
        for x in xrange(0,c1.width):
            for y in xrange(0,c1.height):
                #position = np.dot(c1.KRinv,(c1.pixel[x,y]*np.array([[x,y,1]])+c1.KRT).transpose()).A1
                #coordinates = (np.dot(c2.KR,position)-c2.KRT).transpose().A1
                coordinates = f(c1.KR,c1.KRT,c2.KR,c2.KRT,c1.pixel[x,y]*np.array([[x,y,1]]))[0]
                fill(temp,coordinates[0]/coordinates[2],coordinates[1]/coordinates[2],c1.width,c1.height)
        end = time.time()
        print("DIBR in {}ms".format(end-start))
        return tempImage

class DIBRCamera(Camera):
    def __init__(self,id,settings):
        Camera.__init__(self,id,settings)
        self.referenceViews = []
        self.DIBR_method = DIBR.ImageWarp

    def addReference(self,cam):
        self.referenceViews.append(cam)

    def setReference(self,cam):
        self.referenceViews = [cam]

    def render(self,filename):
        if len(self.referenceViews) == 1:
            tempImage = self.DIBR_method(self.referenceViews[0],self)
            tempImage.save(filename)

cameras = [Camera(id,settings) for id,settings in enumerate(camera_settings)]
for c2 in cameras:
    dibrCam = DIBRCamera(c2.id,c2.settings)
    for c1 in cameras:
        dibrCam.setReference(c1)
        if c1!=c2 and abs(c1.x - c2.x)<2 and abs(c1.y - c2.y)<2:
            dibrCam.render("dibr-simple-results/intersection_{}_{}.png".format(c1.id,c2.id))
            break
    break
