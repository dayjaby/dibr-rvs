import numpy as np
import json
from PIL import Image
import os
import OpenEXR
import Imath
import time
import math

depthPath = 'depth'

with open('cameraSettings.json') as file:
    camera_settings = json.load(file)

cameraCount = len(camera_settings)
cx = [camera['x'] for camera in camera_settings]
cy = [camera['y'] for camera in camera_settings]
P = projection = [np.matrix(camera['projection']) for camera in camera_settings]
K = kalibration = [np.matrix(camera['kalibration']) for camera in camera_settings]
R = rotation = [np.matrix(camera['rotation']) for camera in camera_settings]
T = translation = [np.array(camera['translation']) for camera in camera_settings]
KR = [np.dot(k,r) for k,r in zip(kalibration,rotation)]
KRinv = [np.linalg.inv(kr) for kr in KR]
KRC = [np.dot(kr,t) for kr,t in zip(KR,T)]
files = [camera['file'] for camera in camera_settings]
imgEXR = [OpenEXR.InputFile(os.path.join(depthPath,f+".exr")) for f in files]
dataWindow = [i.header()['dataWindow'] for i in imgEXR]
size = [(dw.max.x-dw.min.x+1,dw.max.y-dw.min.y+1) for dw in dataWindow]
img = [Image.fromstring("F",s,iEXR.channel('R',Imath.PixelType(Imath.PixelType.FLOAT)))
    for s,iEXR in zip(size,imgEXR)]
pix = [i.load() for i in img]
width, height = zip(*size)

def fill(img,x,y,width,height):
    for xf in [math.floor,math.ceil]:
        for yf in [math.floor,math.ceil]:
            wx = xf(x)
            wy = yf(y)
            if wx>=0 and wx<width and wy>=0 and wy<height:
                img[wx,wy] = 1

for c1 in range(cameraCount):
    for c2 in range(cameraCount):
        if c1!=c2 and abs(cx[c1]-cx[c2])<2 and abs(cy[c1]-cy[c2])<2:
            start = time.time()
            tempImage = Image.new("1",(width[c1],height[c1]),"black")
            temp = tempImage.load()
            for x in xrange(0,width[c1]):
                for y in xrange(0,height[c1]):
                    position = np.dot(KRinv[c1],(pix[c1][x,y]*np.array([[x,y,1]])+KRC[c1]).transpose()).A1
                    coordinates = (np.dot(KR[c2],position)-KRC[c2]).transpose().A1
                    fill(temp,coordinates[0]/coordinates[2],coordinates[1]/coordinates[2],width[c1],height[c1])
            end = time.time()
            print "DIBR in {}ms".format(end-start)
            tempImage.save("dibr-simple-results/intersection_{}_{}.png".format(c1,c2))
