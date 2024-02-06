# Lab 1 : Face Detection on Live Video Stream
# Problem Statement
# Connect a video capture device to your laptop or computer.

## Importing warning package to ignore the warnings¶
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

## Load required libraries
import cv2

## Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

## To use a video file as input
cap = cv2.VideoCapture(0)

## Continual loop with face detection task
while True:

    _,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(300) & 0xff
    if k==27:
        break

## Release the VideoCapture object
cv2.destroyAllWindows()
cap.release()