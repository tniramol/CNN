import cv2
import numpy as np

image = cv2.imread('IMP1.jpg')
cv2.imshow('Image', image)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 218])
upper = np.array([157, 54, 255])
mask = cv2.inRange(hsv, lower, upper)