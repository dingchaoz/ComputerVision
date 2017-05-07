import cv2
import sys
import time
import numpy as np
import datetime
# import keras
# from keras.models import load_model
from predict_gender import *


os.chdir('/Users/ejlq/Documents/dingchao/ComputerVision')

# Load the pre-trained face and eye classifier xml file, which are stored in opencv/data/haarcascades/folder
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
numSavedImgs = 0
modelPath = 'saved_model_20170504.h5'
strings = time.strftime("%Y,%m,%d,%H,%M,%S")
newpath = strings.replace(',','') + '/'
os.makedirs(newpath)
modelRan = -1
numFaces = 0
font = cv2.FONT_HERSHEY_SIMPLEX


def read2Gray(imgPath = 'trump_melania.jpg'):
	# Read image and make it in grayscale mode
	img = cv2.imread(imgPath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	return img,gray


def detectFaces(gray,scaleFactor = 1.1,minNeighbors=5,minSize=(30,30)):
	faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize)
	return faces


def saveFaceImg(x,y,w,h,numFaces):
	margin = max(w,h)/2
	cropFace = img[y-margin:y+h+margin,x-margin:x+w+margin]
	saveFName = newpath+'roi'+str(numFaces)+'.png'
	cv2.imwrite(saveFName,cropFace)
	print ('saved face img',numFaces)
	return cropFace,saveFName

def estSex(saveFName):
	# resized_img = resize(cropFace)
	# arr = im2Array(resized_img)
	arr = np.array([loadImg2Array(saveFName)])
	sex = predict_vgg(model,arr)
	return sex

def labelFaces(x,y,w,h):
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	imgLabel = sex
	cv2.putText(img,imgLabel,(x, y), font, 1,(0,255,0),1,cv2.LINE_AA)

model = loadVGG()


while True:
    # Capture frame-by-frame
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detectFaces(gray)


    for (x,y,w,h) in faces:
    	numFaces += 1
    	cropFace,saveFName = saveFaceImg(x,y,w,h,numFaces)
    	if os.stat(saveFName).st_size > 1e5:
    		sex = estSex(saveFName)
    		labelFaces(x,y,w,h)
    	else:
    		os.remove(saveFName)

	# show image with rectangular             
	cv2.imshow('Video',img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


