import cv2
import math
import numpy as np
import imutils
import OpenEXR
import Imath
from PIL import Image
from skimage.color import rgb2yiq, rgb2gray,yiq2rgb
from phasepack import phasecong
import matplotlib.pyplot as plt

def cut_array2d(array, shape):
    arr_shape = np.shape(array)
    xcut = np.linspace(0,arr_shape[0],shape[0]+1).astype(np.int)
    ycut = np.linspace(0,arr_shape[1],shape[1]+1).astype(np.int)
    blocks = [];    xextent = [];    yextent = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            blocks.append(array[xcut[i]:xcut[i+1],ycut[j]:ycut[j+1]])
            xextent.append([xcut[i],xcut[i+1]])
            yextent.append([ycut[j],ycut[j+1]])
    return xextent,yextent,blocks

def makeBlocks(img,blockSizeR,blockSizeC):
    rows, columns = img.shape
    wholeBlockRows = int(math.floor(rows / blockSizeR))
    blockVectorR = blockSizeR * np.ones([1, wholeBlockRows])

    wholeBlockCols = int(math.floor(columns / blockSizeC))
    blockVectorC = blockSizeC * np.ones([1, wholeBlockCols])
    img2 = img.reshape(wholeBlockRows,rows/wholeBlockRows,wholeBlockCols,columns/wholeBlockCols)
    img2 = np.transpose(img2,(0,2,1,3))
    return img2

def dsqm(img1,original,blkSize = 100):
    img1 = img1.astype(np.float32) / 256.0
    original = original.astype(np.float32) / 256.0

    h,w,p = original.shape
    imgL = original
    imgV = img1
    cp = np.zeros((h,w))
    offsetX = 2
    offsetY = 2
    imgLYIQ = rgb2yiq(imgL).astype(np.float32)
    imgLY = imgLYIQ[:,:,1]

    imgVYIQ = rgb2yiq(imgV).astype(np.float32)
    imgVY = imgVYIQ[:,:,1]
    
    brow = blkSize
    bcol = brow
    blkV = makeBlocks(imgVY, brow, bcol)
    blkRows, blkCols = blkV.shape[0:2]
    bestMatch = np.full((blkRows,blkCols), {'v':None,'x':None,'y':None})
    blkVmatch = np.full((blkRows,blkCols), {})
    score = np.zeros(blkV.shape)
    for i in xrange(blkCols):
        for j in xrange(blkRows):
            T = blkV[j,i]
            Tx = i * bcol
            Ty = j * brow
            Bx = (i+1) * bcol
            By = (j+1) * brow

            img = imgLY[max(0, Ty-offsetY):min(h, By+offsetY),max(0,Tx-offsetX):min(w, Bx+offsetX)]
            orig = original[max(0, Ty-offsetY):min(h, By+offsetY),max(0,Tx-offsetX):min(w, Bx+offsetX)]
            warped = imgV[max(0, Ty-offsetY):min(h, By+offsetY),max(0,Tx-offsetX):min(w, Bx+offsetX)]
            b = imgV[j*bcol:(j+1)*bcol,i*brow:(i+1)*brow]
            mp = cv2.matchTemplate(orig, b, cv2.TM_CCORR_NORMED)
            #mp = cv2.matchTemplate(img, T, cv2.TM_CCORR_NORMED)
            y,x = np.unravel_index(np.argmax(mp),mp.shape)
            bestMatch[j,i] = {'v': mp[y,x], 'x': x, 'y': y}
            blkVmatch[j,i]['arr'] = img[y:(y+bcol),x:(x+brow)]
            a = orig[y:y+bcol,x:x+brow]
            x = phasecong(T)
            y = phasecong(blkVmatch[j,i]['arr'])
            score[j,i] = np.abs(x[0].mean()-y[0].mean())
            cp[j*bcol:(j+1)*bcol,i*brow:(i+1)*brow] = (y[0]-x[0])**2*256
    return cp.mean(), cp
#kernel = np.ones((3,3), np.uint8)
#cp = cv2.dilate(cp, kernel, iterations=1)
#cv2.imwrite('x.png',cp)
