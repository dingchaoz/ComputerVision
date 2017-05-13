import cv2
import sys
import time


from predict_gender import *
from PIL import Image

# Load the pre-trained face and eye classifier xml file, which are stored in opencv/data/haarcascades/folder

"""
Usage: python Face_detection.py trump_melania.jpg
"""


def read2Gray(imgPath = 'selfie3.png'):
	# Read image and make it in grayscale mode
	img = cv2.imread(imgPath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	return img,gray


def detectFaces(face_cascade,gray,scaleFactor = 1.1,minNeighbors=5,minSize=(30,30)):
	faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize)
	return faces


def saveFaceImg(x,y,w,h,numFaces,img,newpath):
	margin = int(max(w,h)/2)
	cropFace = img[y-margin:y+h+margin,x-margin:x+w+margin]
	saveFName = newpath+'roi'+str(numFaces)+'.png'
	cv2.imwrite(saveFName,cropFace)
	print ('saved face img',numFaces)
	return cropFace,saveFName

def estSex(saveFName,model):
	# resized_img = resize(cropFace)
	# arr = im2Array(resized_img)
	arr = np.array([loadImg2Array(saveFName)])
	sex = predict_vgg(model,arr)
	return sex

def labelFaces(x,y,w,h,img,sex,font):
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	imgLabel = sex
	cv2.putText(img,imgLabel,(x, y), font, 1,(0,255,0),1,cv2.LINE_AA)


def main(argv):


	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	strings = time.strftime("%Y,%m,%d,%H,%M,%S")
	newpath = strings.replace(',','') + '/'
	os.makedirs(newpath)
	numFaces = 0
	font = cv2.FONT_HERSHEY_SIMPLEX

	model = loadVGG()
	img,gray = read2Gray(str(argv[0]))
	faces = detectFaces(face_cascade,gray)


	for (x,y,w,h) in faces:
		numFaces += 1
		cropFace,saveFName = saveFaceImg(x,y,w,h,numFaces,img,newpath)
		if (os.stat(saveFName).st_size) > 0:
			i_w,i_h = Image.open(saveFName).size
			if 1.2 >i_w/i_h > 0.8:
				sex = estSex(saveFName,model)
				labelFaces(x,y,w,h,img,sex,font)
		else:
			os.remove(saveFName)
		

	# show image with rectangular             
	cv2.imshow('img',img)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main(sys.argv[1:])

