# Lab 3 : Canny edge detection on Live Video Stream
# Problem Statement
# Connect a video capture device to your laptop or computer.

## Importing warning package to ignore the warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

## Load required libraries
import cv2

## To use a video file as input
cap = cv2.VideoCapture(0)

## Setting parameter values
t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

## Continual loop with face detection task
while True:

    _,img = cap.read()
    edge = cv2.Canny(img, t_lower, t_upper)
    cv2.imshow('original', img)
    cv2.imshow('edge', edge)
    k = cv2.waitKey(300) & 0xff
    if k==27:
        break

## Release the VideoCapture object
cv2.destroyAllWindows()
cap.release()
 