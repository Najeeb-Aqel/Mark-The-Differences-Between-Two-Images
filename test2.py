from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import numpy as np
 
# load the two input images
imageA = cv2.imread("input-left.jpg")
imageB = cv2.imread("input-right.jpg")
cv2.imshow("Or", imageA)
 
# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
i = 0
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cooX = (x+(w/2))
	cooY = (y-(h/2))
	print("coordination("+str(i)+") ="+str(cooX)+","+str(cooY))
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
	i = i + 1
 
# show the output images
cv2.imshow("input-left", imageA)
cv2.imshow("input-right", imageB)
cv2.imshow("Differential Gray Scale", diff)
cv2.imshow("Differential Binary Scale", thresh)
cv2.waitKey(0)
