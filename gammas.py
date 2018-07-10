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
from skimage.measure import compare_ssim, compare_mse

import sys
sys.path.append(os.getcwd())
from common import dibr, dsqm
from common.camera import Camera, DIBRCamera
from enum import Enum

depthPath = 'depth'
imgPath = 'img'


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply DIBR on a set of RGBD data')
    parser.add_argument('--rig',type=open)
    parser.add_argument('--dir')
    parser.add_argument('--rgb')
    parser.add_argument('--depth')
    parser.add_argument('--out')
    parser.add_argument('-k',type=int,default=2)
    parser.add_argument('-m',type=int)
    parser.add_argument('--method',choices=["dibr","dibr-simplified","optflow-depth","dsqm","mse-approx"],default='dibr')
    parser.add_argument('--outfile')
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()
    if args.outfile is None:
        args.outfile = sys.stdout
    elif args.append:
        gamma = {}
        k = None

        infile =  open(args.outfile)
        for line in infile.read().splitlines():
            if len(line)>0:
                v = line.split(' ')
                if k is None:
                    k = len(v)/2 - 2
                f = v[0]
                i = 1
                for x in xrange(k+1):
                    y,x = int(v[i]), int(v[i+1])
                    i += 2
                idx = tuple([int(x) for x in v[1:i]])
                if f not in gamma: 
                    gamma[f] = {}
                gamma[f][idx] = float(v[i])
        infile.close()
        args.outfile = open(args.outfile,'a')
    else:
        args.outfile = open(args.outfile,'w+')
    if args.dir is not None:
        if args.rig is None:
            args.rig = open(os.path.join(args.dir,'cameraSettings.json'))
        if args.rgb is None:
            args.rgb = args.dir
        if args.depth is None:
            args.depth = args.dir
    if args.rig is not None:
        s = json.load(args.rig)
        x0, xs = s["xs"]
        y0, ys = s["ys"]
        xvec = np.array(s['translation_x'])
        yvec = np.array(s['translation_y'])
        
        def idx(x,y):
            #return y+x*ys
            return x+y*xs
        print([idx(x,0) for x in xrange(xs)])
        
        cameras = {idx(x,y): Camera(idx(x,y),{
            'x':x,'y':y,
            'distortion':s['distortion'] if 'distortion' in s else None,
            'distortion-depth':s['distortion-depth'] if 'distortion-depth' in s else None,
            'kalibration':s['kalibration'],
            'kalibration-depth':s['kalibration-depth'] if 'kalibration-depth' in s else s['kalibration'],
            'rotation':s['rotation'],
            'translation':np.array(s['translation']) + x*xvec + y*yvec,
            'img_file':s['img_file'].format(s['camera_id'],idx(x,y)),
            'img_directory':args.rgb,
            'depth_file':s['depth_file'].format(s['camera_id'],idx(x,y)),
            'depth_directory':args.depth
        }) for x in xrange(xs) for y in xrange(ys)}
else:
    with open('cameraSettings.json') as file:
        camera_settings = json.load(file)
        cameras = {id: Camera(id,settings) for id,settings in enumerate(camera_settings)}

xyCamera = {(c.x,c.y) : c for id,c in cameras.items()}

n, m, k = xs, args.m, args.k
dibrCams = {}
for y in xrange(0,ys):
    if k == 2:
        for l in xrange(0,xs-2):
            if m is None or l<m-1:
                rng = xrange(l+2,n)
            else:
                rng = xrange(l+2,n+l-m+2)
            for r in rng:
                for j in xrange(l+1,r):
                    if args.method.startswith("dibr"):
                        if (y,j) in dibrCams:
                            dibrCam = dibrCams[(y,j)]
                        else:
                            dibrCam = DIBRCamera(xyCamera[(j,y)].id, xyCamera[(j,y)].settings)
                            dibrCams[(y,j)] = dibrCam
                        dibrCam.setReferences([xyCamera[(l,y)],xyCamera[(r,y)]])
                        if args.method == "dibr":
                            if not args.append or 'mse' not in gamma or (y,j,y,l,y,r) not in gamma['mse'] or 'ssim' not in gamma or (y,j,y,l,y,r) not in gamma['ssim']\
                             or 'depthmse' not in gamma or (y,j,y,l,y,r) not in gamma['depthmse'] or 'depthssim' not in gamma or (y,j,y,l,y,r) not in gamma['depthssim']:
                                dibrCam.render(os.path.join(args.out,"dibr_{}_{}_{}_{}.png".format(y,j,l,r)),os.path.join(args.out,"dibr_{}_{}_{}_{}.exr".format(y,j,l,r)))
                                original = np.array(xyCamera[(j,y)].colorPixel)
                                synthetic = np.array(dibrCam.img.getdata(),dtype=np.uint8).reshape(original.shape)
                                mse = compare_mse(original,synthetic)/65536.0
                                ssim = compare_ssim(original,synthetic,multichannel=True,gaussian_weights=True,use_sample_covariance=False,sigma=1.5)
                                ssim = 1-(1+ssim)/2
                                args.outfile.write("{} {} {} {} {} {} {} {}\n".format('mse',y,j,y,l,y,r,mse))
                                args.outfile.write("{} {} {} {} {} {} {} {}\n".format('ssim',y,j,y,l,y,r,ssim))
                                original = np.array(xyCamera[(j,y)].depthPixel)
                                synthetic = np.array(dibrCam.depth.getdata(),dtype=np.float32).reshape(original.shape)
                                mse = (1 - 1/(1 + (original - synthetic)**2)).mean()
                                #mse = compare_mse(original,synthetic)/65536.0
                                ssim = compare_ssim(original,synthetic,gaussian_weights=True,use_sample_covariance=False,sigma=1.5)
                                ssim = 1-(1+ssim)/2
                                args.outfile.write("{} {} {} {} {} {} {} {}\n".format('depthmse',y,j,y,l,y,r,mse))
                                args.outfile.write("{} {} {} {} {} {} {} {}\n".format('depthssim',y,j,y,l,y,r,ssim))
                        elif args.method == "dibr-simplified":
                            if not args.append or 'smse' not in gamma or (y,j,y,l,y,r) not in gamma['smse'] or 'sssim' not in gamma or (y,j,y,l,y,r) not in gamma['sssim']:
                                dibrCam.DIBR_method = dibr.InverseMapping2Simplified
                                dibrCam.render(os.path.join(args.out,"dibrs_{}_{}_{}_{}.png".format(y,j,l,r)))
                                original = np.array(xyCamera[(j,y)].colorPixel)
                                synthetic = np.array(dibrCam.img.getdata(),dtype=np.uint8).reshape(original.shape)
                                mse = compare_mse(original,synthetic)/65536.0
                                ssim = compare_ssim(original,synthetic,multichannel=True,gaussian_weights=True,use_sample_covariance=False,sigma=1.5)
                                ssim = 1-(1+ssim)/2
                                args.outfile.write("{} {} {} {} {} {} {} {}\n".format('smse',y,j,y,l,y,r,mse))
                                args.outfile.write("{} {} {} {} {} {} {} {}\n".format('sssim',y,j,y,l,y,r,ssim))
                    elif args.method == "dsqm":
                        scale = 4
                        if not args.append or 'dsqm' not in gamma or (y,j,y,l,y,r) not in gamma['dsqm']:
                            original = np.array(xyCamera[(j,y)].colorPixel)[:,:,0:3]
                            original = cv2.resize(original,None,fx=1.0/scale,fy=1.0/scale,interpolation=cv2.INTER_AREA)
                            synthetic = cv2.imread(os.path.join(args.out,"dibr_{}_{}_{}_{}.png".format(y,j,l,r)))[:,:,[2,1,0]]
                            synthetic = cv2.resize(synthetic,None,fx=1.0/scale,fy=1.0/scale,interpolation=cv2.INTER_AREA)
                            score, pc = dsqm.dsqm(synthetic,original,100/scale)
                            #cv2.imwrite(os.path.join("blender_output_dsqm","dsqm_{}_{}_{}_{}_1.png".format(y,j,l,r)),original)
                            #cv2.imwrite(os.path.join("blender_output_dsqm","dsqm_{}_{}_{}_{}_2.png".format(y,j,l,r)),synthetic)
                            cv2.imwrite(os.path.join("blender_output_dsqm","dsqm_{}_{}_{}_{}.png".format(y,j,l,r)),pc)
                            args.outfile.write("{} {} {} {} {} {} {} {}\n".format('dsqm',y,j,y,l,y,r,score))
                    elif args.method == "mse-approx":
                        if not args.append or 'mseapprox' not in gamma or (y,j,y,l,y,r) not in gamma['mseapprox']:
                            original = np.array(xyCamera[(j,y)].colorPixel)[:,:,0:3]
                            synthetic1 = cv2.imread(os.path.join(args.out,"dibr_{}_{}_{}.png".format(y,j,r)))[:,:,[2,1,0]]
                            synthetic2 = cv2.imread(os.path.join(args.out,"dibr_{}_{}_{}.png".format(y,j,l)))[:,:,[2,1,0]]
                            mse1 = ((original - synthetic1) ** 2)
                            mse2 = ((original - synthetic2) ** 2)
                            mse = np.minimum(mse1,mse2)
                            synthetic = synthetic1
                            eq2 = (mse == mse2)
                            synthetic[eq2] = synthetic2[eq2]
                            mse = mse.mean()/65536.0
                            ssim = compare_ssim(original,synthetic,multichannel=True,gaussian_weights=True,use_sample_covariance=False,sigma=1.5)
                            ssim = 1-(1+ssim)/2
                            args.outfile.write("{} {} {} {} {} {} {} {}\n".format('mseapprox',y,j,y,l,y,r,mse))
                            args.outfile.write("{} {} {} {} {} {} {} {}\n".format('ssimapprox',y,j,y,l,y,r,ssim))

                    elif args.method == "optflow-depth":
                        if not args.append or 'optflow-depth' not in gamma or (y,j,y,l,y,r) not in gamma['optflow-depth']:
                            flow_left = cv2.calcOpticalFlowFarneback(xyCamera[(l,y)].depthPixel,xyCamera[(j,y)].depthPixel, 0.5, 3, 15, 3, 5, 1.1, 0)
                            flow_right = cv2.calcOpticalFlowFarneback(xyCamera[(r,y)].depthPixel,xyCamera[(j,y)].depthPixel, 0.5, 3, 15, 3, 5, 1.1, 0)
                            flow = np.minimum(np.linalg.norm(flow_left.reshape(-1,2),axis=1),np.linalg.norm(flow_right.reshape(-1,2),axis=1))
                            flow = np.mean(flow/(flow+1))
                            args.outfile.write("{} {} {} {} {} {} {} {}\n".format('optflow-depth',y,j,y,l,y,r,flow))
                            
    elif k == 1:
        for r in xrange(0,n):
            for j in xrange(0,n):
                if j != r:
                    if args.method.startswith("dibr"):
                        if (y,j) in dibrCams:
                            dibrCam = dibrCams[(y,j)]
                        else:
                            dibrCam = DIBRCamera(xyCamera[(j,y)].id, xyCamera[(j,y)].settings)
                            dibrCams[(y,j)] = dibrCam
                        dibrCam.setReferences([xyCamera[(r,y)]])
                        if args.method == "dibr":
                            if not args.append or 'mse' not in gamma or (y,j,y,r) not in gamma['mse'] or 'ssim' not in gamma or (y,j,y,r) not in gamma['ssim']\
                             or 'depthmse' not in gamma or (y,j,y,l,y,r) not in gamma['depthmse'] or 'depthssim' not in gamma or (y,j,y,r) not in gamma['depthssim']:
                                dibrCam.render(os.path.join(args.out,"dibr_{}_{}_{}.png".format(y,j,r)),os.path.join(args.out,"dibr_{}_{}_{}.exr".format(y,j,r)))
                                original = np.array(xyCamera[(j,y)].colorPixel)
                                synthetic = np.array(dibrCam.img.getdata(),dtype=np.uint8).reshape(original.shape)
                                mse = compare_mse(original,synthetic)/65536.0
                                ssim = compare_ssim(original,synthetic,multichannel=True,gaussian_weights=True,use_sample_covariance=False,sigma=1.5)
                                ssim = 1-(1+ssim)/2
                                args.outfile.write("{} {} {} {} {} {}\n".format('mse',y,j,y,r,mse))
                                args.outfile.write("{} {} {} {} {} {}\n".format('ssim',y,j,y,r,ssim))
                                original = np.array(xyCamera[(j,y)].depthPixel)
                                synthetic = np.array(dibrCam.depth.getdata(),dtype=np.float32).reshape(original.shape)
                                mse = (1 - 1/(1 + (original - synthetic)**2)).mean()
                                #mse = compare_mse(original,synthetic)/65536.0
                                ssim = compare_ssim(original,synthetic,gaussian_weights=True,use_sample_covariance=False,sigma=1.5)
                                ssim = 1-(1+ssim)/2
                                args.outfile.write("{} {} {} {} {} {}\n".format('depthmse',y,j,y,r,mse))
                                args.outfile.write("{} {} {} {} {} {}\n".format('depthssim',y,j,y,r,ssim))
                        elif args.method == "dibr-simplified":
                            if not args.append or 'smse' not in gamma or (y,j,y,r) not in gamma['smse'] or 'sssim' not in gamma or (y,j,y,r) not in gamma['sssim']:
                                dibrCam.DIBR_method = dibr.InverseMapping2Simplified
                                dibrCam.render(os.path.join(args.out,"dibrs_{}_{}_{}.png".format(y,j,r)))
                                original = np.array(xyCamera[(j,y)].colorPixel)
                                synthetic = np.array(dibrCam.img.getdata(),dtype=np.uint8).reshape(original.shape)
                                mse = compare_mse(original,synthetic)/65536.0
                                ssim = compare_ssim(original,synthetic,multichannel=True,gaussian_weights=True,use_sample_covariance=False,sigma=1.5)
                                ssim = 1-(1+ssim)/2
                                args.outfile.write("{} {} {} {} {} {}\n".format('smse',y,j,y,r,mse))
                                args.outfile.write("{} {} {} {} {} {}\n".format('sssim',y,j,y,r,ssim))
                    elif args.method == "dsqm":
                        scale = 4
                        if not args.append or 'dsqm' not in gamma or (y,j,y,r) not in gamma['dsqm']:
                            original = np.array(xyCamera[(j,y)].colorPixel)[:,:,0:3]
                            original = cv2.resize(original,None,fx=1.0/scale,fy=1.0/scale,interpolation=cv2.INTER_AREA)
                            synthetic = cv2.imread(os.path.join(args.out,"dibr_{}_{}_{}.png".format(y,j,r)))[:,:,[2,1,0]]
                            synthetic = cv2.resize(synthetic,None,fx=1.0/scale,fy=1.0/scale,interpolation=cv2.INTER_AREA)
                            score, pc = dsqm.dsqm(synthetic,original,100/scale)
                            #cv2.imwrite(os.path.join("blender_output_dsqm","dsqm_{}_{}_{}_{}_1.png".format(y,j,l,r)),original)
                            #cv2.imwrite(os.path.join("blender_output_dsqm","dsqm_{}_{}_{}_{}_2.png".format(y,j,l,r)),synthetic)
                            cv2.imwrite(os.path.join("blender_output_dsqm","dsqm_{}_{}_{}.png".format(y,j,r)),pc)
                            args.outfile.write("{} {} {} {} {} {}\n".format('dsqm',y,j,y,r,score))
                    elif args.method == "optflow-depth":
                        if not args.append or 'optflow-depth' not in gamma or (y,j,y,r) not in gamma['optflow-depth']:
                            flow = cv2.calcOpticalFlowFarneback(xyCamera[(r,y)].depthPixel,xyCamera[(j,y)].depthPixel, 0.5, 3, 15, 3, 5, 1.1, 0)
                            flow = np.linalg.norm(flow.reshape(-1,2),axis=1)
                            flow = np.mean(flow/(flow+1))
                            args.outfile.write("{} {} {} {} {} {}\n".format('optflow-depth',y,j,y,r,flow))