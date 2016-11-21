import numpy as np
import json
from PIL import Image
import os
import OpenEXR
import Imath
import time

depthPath = 'depth'

with open('cameraSettings.json') as file:
    camera_settings = json.load(file)

scale = 1

cameraCount = len(camera_settings)
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

for c1 in range(2):
    for c2 in range(2):
        if c1!=c2:
            start = time.time()
            tempImage = Image.new("1",(width[c1]/scale,height[c1]/scale),"black")
            temp = tempImage.load()
            for x in xrange(0,width[c1],scale):
                for y in xrange(0,height[c1],scale):
                    position = np.dot(KRinv[c1],(pix[c1][x,y]*np.array([[x,y,1]])+KRC[c1]).transpose()).A1
                    temp[x/scale,y/scale] = 0
            end = time.time()
            print "DIBR in {}ms".format(end-start)
            tempImage.save("intersection_{}_{}.png".format(c1,c2))
