# LAB 4 - Average Kernel based blurring a given image

## Load OpenCV & Numpy library
import cv2
import numpy as np

## Read the image
img = cv2.imread('demo_face.jpg')

## Configure blur kernel
kernel2 = np.ones((5, 5), np.float32) / 25

## Applying the blur filter
img2 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)

## Show original and edges image
cv2.imshow('original', img)
cv2.imshow('edge', img2)

## Release the Window object
cv2.waitKey(0)
cv2.destroyAllWindows()