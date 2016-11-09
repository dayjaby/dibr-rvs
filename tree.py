from scipy import spatial
import numpy as np
import json
from PIL import Image
import os
import OpenEXR
import Imath

depthPath = 'depth'

with open('cameraSettings.json') as file:
    camera_settings = json.load(file)

camera_to_indices = dict()
index_to_camera = []
camera_tree = dict()
positions = []
start_indices = dict()
tree_size = dict()
scale = 1


for camera in camera_settings[:2]:
    f = camera['file']
    cam = np.array(camera['camera'])
    P = np.matrix(camera['projection'])
    MV = np.matrix(camera['mv'])
    Pinv = P.getI()
    K = np.matrix(camera['kalibration'])
    R = np.matrix(camera['rotation'])
    C = np.array(camera['translation'])
    KR = np.dot(K,R)
    KRinv = np.linalg.inv(KR)
    KRC = np.dot(KR,C)

    # v = np.dot(P,np.array([-6.88,7.34,-6.211,1.0])).tolist()[0]
    # print (v[0]/v[2],v[1]/v[2])

    imgEXR = OpenEXR.InputFile(os.path.join(depthPath,f+".exr"))
    dw = imgEXR.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redChannel = imgEXR.channel('R',Imath.PixelType(Imath.PixelType.FLOAT))
    img = Image.fromstring("F",size,redChannel)
    pix = img.load()
    width, height = img.size
    img2 = Image.new("1",(img.size[0]/scale,img.size[1]/scale),"black")
    pix2 = img2.load()
    indices = []
    points = []
    start_index = len(positions)

    for x in xrange(0,width,scale):
        for y in xrange(0,height,scale):
            position = np.dot(KRinv,(pix[x,y]*np.array([[x,y,1]])+KRC).transpose()).A1
            cameras = [f]
            for f2, tree in camera_tree.items():
                factor = 1.2*pix[x,y]/width
                distance, index = tree.query([position],distance_upper_bound=factor)
                if index<tree_size[f2]:
                    pix2[x/scale,y/scale] = 0
                    cameras.append(f2)
                else:
                    pix2[x/scale,y/scale] = 1

            idx = len(points) + start_index
            indices.append(idx)
            points.append(position)
            index_to_camera.append(cameras)

    start_indices[f] = start_index
    tree_size[f] = len(points)
    positions.extend(points)
    print points[0]
    print len(points)
    print np.shape(points)
    camera_tree[f] = spatial.KDTree(points)
    camera_to_indices[f] = indices
    img2.save("intersection.png")
