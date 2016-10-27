from scipy import spatial
import numpy as np
import json
from PIL import Image
import os

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
    x0y0 = np.array(camera['top_left'])
    x1y0 = np.array(camera['top_right'])
    x0y1 = np.array(camera['bottom_left'])
    x1y1 = np.array(camera['bottom_right'])
    cam = np.array(camera['camera'])
    depth_factor = camera['depth_factor']
    img = Image.open(os.path.join(depthPath,f))
    pix = img.load()
    width, height = img.size
    img2 = Image.new("1",(img.size[0]/scale,img.size[1]/scale),"black")
    pix2 = img2.load()
    indices = []
    points = []
    start_index = len(positions)

    for x in xrange(0,width,scale):
        top =    x1y0 * x/width + x0y0 * (width-x)/width
        bottom = x1y1 * x/width + x1y0 * (width-x)/width
        for y in xrange(0,height,scale):
            if pix[x,y]*depth_factor < 100:
                interpolation = bottom * y/height + top * (height-y)/height
                diff = interpolation - cam
                position = diff/np.linalg.norm(diff)*pix[x,y]*depth_factor + cam
                dist = np.linalg.norm(position-cam)
                cameras = [f]
                for f2, tree in camera_tree.items():
                    factor = 0.11
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
    camera_tree[f] = spatial.KDTree(points)
    camera_to_indices[f] = indices
    img2.save("intersection.png")
