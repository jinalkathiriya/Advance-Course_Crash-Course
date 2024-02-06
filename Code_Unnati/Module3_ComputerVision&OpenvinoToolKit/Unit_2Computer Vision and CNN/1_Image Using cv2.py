# Lab - 1 Basic Operation on Image Using cv2

##Exercise-1 Image Reading in Different Types
import cv2
import numpy as np
image = cv2.imread("CV.png")

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exercise-2 Starting Video Camera
## Steps to capture a video:
# import the opencv library
import cv2
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

## Exercise-3 Image Filter
image = cv2.imread("CV.png")

# Print error message if image is null
if image is None:
    print('Could not read image')

# Apply identity kernel
kernel1 = np.array([[-7, 0, 7],
                    [-7, 0, 7],
                    [-7, 0, 7]])

identity = cv2.filter2D(src=image,  ddepth=-1,kernel=kernel1)


cv2.imshow('Original', image)
cv2.imshow('Identity', identity)
    
cv2.waitKey()
cv2.imwrite('identity.jpg', identity)
cv2.destroyAllWindows()

cv2.imshow('Original', image)
cv2.imshow('Identity', identity)
    
cv2.waitKey()
cv2.imwrite('identity.jpg', identity)
cv2.destroyAllWindows()

# Apply blurring kernel
kernel2 = np.ones((5, 5), np.float32) / 25
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

cv2.imshow('Original', image)
cv2.imshow('Kernel Blur', img)
    
cv2.waitKey()
cv2.imwrite('blur_kernel.jpg', img)
cv2.destroyAllWindows()

b,g,r=cv2.split(image)

g=b//2

g

img=cv2.merge((b,g,r))

cv2.imshow('Original', img)
cv2.waitKey()
cv2.destroyAllWindows()



