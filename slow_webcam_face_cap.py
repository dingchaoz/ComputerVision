import time
from predict_gender import *
import collections
import numpy as np
from PIL import ImageGrab


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
process_this_frame = True
last3window_avg = collections.deque(maxlen=3)
cur_window=collections.deque(maxlen=5)
facechange_res= collections.deque(maxlen=3)
facenum_when_captured = 0
genders = []


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

def labelFaces(x,y,w,h,gender):
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    imgLabel = gender

    cv2.rectangle(img,(x,y+h+int(h/10)),(x+w,y+h),(0,0,255),cv2.FILLED)

    cv2.putText(img,imgLabel,(x+int(w/3), y+h+int(h/11)), font, 1,(255,255,255),1,1)


def faceNumChange(cur_window,last3window_avg):

	if np.mean(list(cur_window)) > np.mean(list(last3window_avg)):
		res = 'face num increased'
	elif np.mean(list(cur_window)) < np.mean(list(last3window_avg)):
		res = 'face num decreased'
	else:
		res = 'face num not changed'

	return res

def captureNewFace(l,facenum,facnum_when_captured):
	if len(list(l)) == 3:
		if l[0] == l[1] == l[2] == 'face num not changed' and facnum_when_captured != facenum:
			return True

def readScreen():
    screen = ImageGrab.grab(bbox=(0,0,1500,1500)) #bbox specifies specific region (bbox= x,y,width,height)
    #screen = pyscreenshot.grab(bbox=(0,0,1200,900))
    screen_np = np.array(screen)
    gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

    img = cv2.cvtColor(screen_np, cv2.COLOR_RGBA2RGB)

    return screen,gray,img


def faceHistory(faces,cur_window,last3window_avg,faceNumChange,facechange_res):
    faces = detectFaces(gray)
    facenum = len(faces)
    cur_window.append(facenum)
    last3window_avg.append(np.mean(list(cur_window)))
    change = faceNumChange(cur_window,last3window_avg)
    facechange_res.append(change)


    return facechange_res,facenum,change


model = loadVGG()


while True:
    # Capture frame-by-frame
    img,gray = read2Gray()

    #screen,gray,img = readScreen()

    #if process_this_frame:
    faces = detectFaces(gray)
    facechange_res,facenum,change = faceHistory(faces,cur_window,last3window_avg,faceNumChange,facechange_res)


    if captureNewFace(facechange_res,facenum,facenum_when_captured):


        #print ('facenum_when_caputred', facenum_when_captured)



        print ('capture faces')
        print (len(faces))
        if len(faces) == 0:
            genders  = []
            facenum_when_captured = 0

        if len(genders)>= len(faces):
            genders = []

        for (x,y,w,h) in faces:
            numFaces += 1
            cropFace,saveFName = saveFaceImg(x,y,w,h,numFaces)

            if (os.stat(saveFName).st_size) > 0:
                i_w,i_h = Image.open(saveFName).size
                if 1.2 >i_w/i_h > 0.8:
                    gender = estSex(saveFName)
                    facenum_when_captured = facenum
                    if len(genders)> len(faces):
                        genders = []
                    genders.append(gender)
                    print (gender)
                    print (genders)
                    #labelFaces(x,y,w,h)
                else:
                    os.remove(saveFName)
            else:
                os.remove(saveFName)

    #process_this_frame = not process_this_frame

    # TODO: remove faces that are not a real face

    for (x,y,w,h),g in zip(faces,genders):

        labelFaces(x,y,w,h,g)


    # show image with rectangular
    cv2.putText(img, str(facenum) + ' '+ change, (100, 100), font, 1.0, (255, 0, 0), 1)
    cv2.imshow('Video',img)

    # screenshot = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    # cv2.putText(img, str(facenum) + ' '+ change, (100, 100), font, 1.0, (255, 0, 0), 1)
    # cv2.imshow("screenshot", screenshot)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


