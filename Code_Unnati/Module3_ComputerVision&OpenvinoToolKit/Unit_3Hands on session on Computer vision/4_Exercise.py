# Lab 4 : Edge Detection Using Prewitt Operator
# Problem Statement
# Use below matrix as kernel for convolution.
# kernel = [ [-7, 0, 7], [-7, 0, 7], [-7, 0, 7] ]
# You need to implement a filter based edge detection system

## Importing warning package to ignore the warnings¶
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

## Load required libraries
import cv2
import numpy as np

## Read input image
img = cv2.imread('demo_face.jpg')

## Configure Prewitt kernel
kernel1 = np.array([[-7, 0, 7],
                    [-7, 0, 7],
                    [-7, 0, 7]])

## Applying the blur filter
img2 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel1)

## Show original and edges image
cv2.imshow('original', img)
cv2.imshow('edge', img2)

## Release the Window object
cv2.waitKey(0)
cv2.destroyAllWindows()