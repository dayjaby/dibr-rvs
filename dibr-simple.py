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
from optimal_tree import rvs
import argparse

import sys
sys.path.append(os.getcwd())
from common import dibr
from common.camera import Camera, DIBRCamera

depthPath = 'depth'
imgPath = 'img'


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply DIBR on a set of RGBD data')
    parser.add_argument('--rig',type=open)
    parser.add_argument('--rgb')
    parser.add_argument('--depth')
    args = parser.parse_args()
    if args.rig is not None:
        s = json.load(args.rig)
        x0, xs = s["xs"]
        y0, ys = s["ys"]
        print(x0)
        print(xs)
        print(ys)
        xvec = np.array(s['translation_x'])
        yvec = np.array(s['translation_y'])
        cameras = {x+y*xs: Camera(x+y*xs,{
            'x':x,'y':y,
            'kalibration':s['kalibration'],
            'rotation':s['rotation'],
            'translation':np.array(s['translation']) + x*xvec + y*yvec,
            'img_file':s['img_file'].format(s['camera_id'],x+y*xs),
            'img_directory':args.rgb,
            'depth_file':s['depth_file'].format(s['camera_id'],x+y*xs),
            'depth_directory':args.depth
        }) for x in xrange(xs) for y in xrange(ys)}
        print(cameras.items()[0].settings)
else:
    with open('cameraSettings.json') as file:
        camera_settings = json.load(file)
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

#dibrCam = DIBRCamera(xyCamera[(1,0)].id, xyCamera[(1,0)].settings)
#dibrCam.addReference(xyCamera[(0,0)])
#dibrCam.addReference(xyCamera[(2,0)])
#dibrCam.render("dibr-simple-results/dibr_test_2.png","dibr-simple-results/dibr_test_2.exr")
"""for c2id, c2 in cameras.items():
    dibrCam = DIBRCamera(c2.id,c2.settings)
    for c1id, c1 in cameras.items():
        dibrCam.setReference(c1)
        if c1!=c2:
            if abs(c1.x - c2.x)<2 and abs(c1.y - c2.y) < 2:
                dibrCam.render("dibr-simple-results/dibr_{}_{}.png".format(c1.id,c2.id),"dibr-simple-results/intersection_{}_{}.exr".format(c1.id,c2.id))
                break
    break
"""

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
