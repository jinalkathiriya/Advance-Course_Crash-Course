# LAB 1 - Detect faces present in a given image

## Load OpenCV library
import cv2

## Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

## Read the image
img = cv2.imread('demo_face.jpg')

## Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Detect the faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

## Draw the rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

##Show original image
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()