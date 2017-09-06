import cv2
import numpy as np

def draw_flow(im,flow,step=24):
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0),-1)
    return vis
img1 = cv2.imread('img/0000.png')
img2 = cv2.imread('img/0002.png')
prev = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
next = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
flow = cv2.calcOpticalFlowFarneback(prev,next, 0.5, 3, 15, 3, 5, 1.1, 0)
print(flow[0,0])
print(np.mean(np.linalg.norm(flow.reshape(-1,2),axis=1)))
cv2.imshow('image',draw_flow(prev,flow))
cv2.waitKey(0)
cv2.destroyAllWindows()

