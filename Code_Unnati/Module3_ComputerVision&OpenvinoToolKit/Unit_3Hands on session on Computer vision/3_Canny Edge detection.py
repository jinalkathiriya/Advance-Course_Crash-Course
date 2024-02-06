## LAB 3 - Canny Edge detection on a given image

## Load OpenCV library
import cv2

## Read the image
img = cv2.imread('demo_face.jpg')

## Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

## Applying the Canny Edge filter
edge = cv2.Canny(img, t_lower, t_upper)

## Show original and edges image
cv2.imshow('original', img)
cv2.imshow('edge', edge)

## Release the Window object
cv2.waitKey(0)
cv2.destroyAllWindows()
 