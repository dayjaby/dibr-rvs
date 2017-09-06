import dibr
import cv2

pic = cv2.imread("img/0000.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
pic2 = dibr.inverseMapping(pic)
cv2.imshow("Result",pic2)
cv2.waitKey()

