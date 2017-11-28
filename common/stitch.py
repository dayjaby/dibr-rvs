# import the necessary packages
import numpy as np
import imutils
import cv2
 
class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
 
		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)
 
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None
        
		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0] + imageB.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
 
		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)
 
			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)
 
		# return the stitched image
		return result
    
	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)
 
		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)
 
			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)
 
		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])
 
		# return a tuple of keypoints and features
		return (kps, features)
    
	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)
 
		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)
 
			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)
 
		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])
 
		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
 
		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
 
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
 
			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)
 
		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
 
		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
 
		# return the visualization
		return vis

if __name__ == "__main__":
    import argparse, time
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", required=True,
        help="path to the first image")
    ap.add_argument("-s", "--second", required=True,
        help="path to the second image")
    ap.add_argument("-x", nargs=3, type=int, help="x coordinate of the first image, second image, combined image")
    ap.add_argument("-y", nargs=3, type=int, help="y coordinate of the first image, second image, combined image")
    args = ap.parse_args()
    # load the two images and resize them to have a width of 400 pixels
    # (for faster processing)
    imageA = cv2.imread(args.first)
    imageB = cv2.imread(args.second)
    original_height, original_width, _ = imageA.shape
    #imageB = imutils.resize(imageB, width=400)

    # stitch the images together to create a panorama
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    print(result.shape)
    
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    
    along_height = np.sum(result,axis=(0,2))
    res = consecutive(np.where(along_height==0))[-1]
    if len(res[0]) == 0:
        width = result.shape[1]
    else:
        width = res[0,0]+1
    along_width = np.sum(result,axis=(1,2))
    res = consecutive(np.where(along_width==0))[-1]
    if len(res[0]) == 0:
        height = result.shape[0]
    else:
        height = res[0,0]+1
    result = result[0:height,0:width]
    #print(result.shape)
    delta_x = (width-original_width)/float(args.x[1]-args.x[0])
    delta_y = (height-original_height)/float(args.y[1]-args.y[0])
    rx = int((args.x[2]-args.x[0])*delta_x)
    ry = int((args.y[2]-args.y[0])*delta_y)
    result = result[ry:ry+original_height,rx:rx+original_width]
    print(rx,ry)
    # show the images
    #cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    #cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    ch = 0
    while  ch != 1048603:
        ch = cv2.waitKey(33)
        if ch!=-1:
            print(ch)
        time.sleep(0.01)
    
        