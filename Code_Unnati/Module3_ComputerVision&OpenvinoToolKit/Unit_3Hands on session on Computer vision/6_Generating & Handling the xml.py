# LAB 6 - Generating & Handling the .xml files

## Load OpenCV & Numpy library
import cv2
import numpy as np

## Create a new filter
filter_kernel = np.array([[-1, 0, +1],
                          [-2, 0, +2],
                          [-1, 0, +1]], dtype=np.float32)

## Save the filter to an XML file
cv2.FileStorage("filter.xml", cv2.FILE_STORAGE_WRITE).write("filter", filter_kernel)
 

## Now lets use the created .xml file
## Load the filter from the XML file
file_storage = cv2.FileStorage("filter.xml", cv2.FILE_STORAGE_READ)
filter_kernel = file_storage.getNode("filter").mat()

## Load an image
image = cv2.imread("test_image.jpg")

## Apply the filter to the image
filtered_image = cv2.filter2D(image, -1, filter_kernel)

## Display the original and filtered images
cv2.imshow("Original Image", image)
cv2.imshow("Filtered Image", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()