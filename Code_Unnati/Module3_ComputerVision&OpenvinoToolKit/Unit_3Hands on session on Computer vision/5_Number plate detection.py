# LAB 5 - Number plate detection system

## Load OpenCV & Numpy library
import cv2
import numpy as np

## Configure cascade classifier
np_cascade=cv2.CascadeClassifier('np.xml')

## To capture video from webcam.
cap = cv2.VideoCapture(0)

## Continual loop to detect number plates
while True:
    _,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = np_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(300) & 0xff
    if k==27:
        break

## Release the VideoCapture object
cv2.destroyAllWindows()

cap.release()