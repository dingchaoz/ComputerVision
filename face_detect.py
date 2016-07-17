# Reference
# https://realpython.com/blog/python/face-recognition-with-python/

import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
# Things learnt with the parameters
# scaleFactor, the smaller the face in the pic(the further the camera was away), 
# the higher the scale Factor should be; the opposite applies to minSize
# faces = faceCascade.detectMultiScale(
#     gray,
#     scaleFactor=1.07,
#     minNeighbors=5,
#     minSize=(130, 130)
#     #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
# )


faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.07,
    minNeighbors=5,
    minSize=(100, 100)
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)