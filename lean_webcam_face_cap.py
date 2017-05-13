import cv2
import sys
import time
import numpy as np
import datetime

from predict_gender import *



# Load the pre-trained face and eye classifier xml file, which are stored in opencv/data/haarcascades/folder
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
numSavedImgs = 0
strings = time.strftime("%Y,%m,%d,%H,%M,%S")
newpath = strings.replace(',','') + '/'
os.makedirs(newpath)
modelRan = -1
numFaces = 0
font = cv2.FONT_HERSHEY_SIMPLEX


def read2Gray():
	# Read image and make it in grayscale mode
	ret, img = video_capture.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img,gray


def detectFaces(gray,scaleFactor = 1.1,minNeighbors=5,minSize=(30,30)):
	faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize)
	return faces


def saveFaceImg(x,y,w,h,numFaces):
	margin = int(max(w,h)/2)
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
    img,gray = read2Gray()
    faces = detectFaces(gray)

    '''
    Drawing facial landmark
    Save 10 pics initially
    Analyze facial measurements and give it a index number
    Predict gender and assign a label


    If face number changes up or down for more than 5 secs
    Compare if detected faces are existing faces,
    if yes:
            just give the pre-estiamted label
    if not:
            draw facial landmark, save 10 pics,
            analyze measurement, predict gender and assign a label


    '''

    for (x,y,w,h) in faces:
        numFaces += 1
        cropFace,saveFName = saveFaceImg(x,y,w,h,numFaces)

        if (os.stat(saveFName).st_size) > 0:
            i_w,i_h = Image.open(saveFName).size
            if 1.2 >i_w/i_h > 0.8:
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


