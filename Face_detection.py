import cv2
import sys
import time
import numpy as np
import datetime
# import keras
# from keras.models import load_model
from predict_gender import *

# Load the pre-trained face and eye classifier xml file, which are stored in opencv/data/haarcascades/folder
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
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
img,gray = read2Gray()
faces = detectFaces(gray)



for (x,y,w,h) in faces:
	numFaces += 1
	cropFace,saveFName = saveFaceImg(x,y,w,h,numFaces)
	sex = estSex(saveFName)
	labelFaces(x,y,w,h)
	

	
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    #cv2.putText(img,'l1',(x+w, y+h), font, 1,(0,255,0),1,cv2.LINE_AA)

# show image with rectangular             
cv2.imshow('img',img)
cv2.waitKey(10000)
cv2.destroyAllWindows()





# model = loadVGG(modelPath)


model = loadVGG()
