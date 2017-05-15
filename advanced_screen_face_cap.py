from PIL import ImageGrab
import time
from predict_gender import *
import face_recognition


#os.chdir('/Users/ejlq/Documents/dingchao/ComputerVision')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
numSavedImgs = 0
strings = time.strftime("%Y,%m,%d,%H,%M,%S")
newpath = strings.replace(',','') + '/'
os.makedirs(newpath)
modelRan = -1
numFaces = 0
font = cv2.FONT_HERSHEY_SIMPLEX
process_this_frame = True

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

def encodeFace(saveFNAME):
    image = face_recognition.load_image_file(saveFNAME)
    face_encoding = face_recognition.face_encodings(image)[0]

    return face_encoding

model = loadVGG()


def readScreen():
    screen = ImageGrab.grab(bbox=(0,0,1500,1500)) #bbox specifies specific region (bbox= x,y,width,height)
    #screen = pyscreenshot.grab(bbox=(0,0,1200,900))
    screen_np = np.array(screen)
    gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

    img = cv2.cvtColor(screen_np, cv2.COLOR_RGBA2RGB)

    return screen,gray,img

while(True):
    # screen = ImageGrab.grab(bbox=(0,0,1500,1500)) #bbox specifies specific region (bbox= x,y,width,height)
    # #screen = pyscreenshot.grab(bbox=(0,0,1200,900))
    # screen_np = np.array(screen)
    # gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)
    #
    # img = cv2.cvtColor(screen_np, cv2.COLOR_RGBA2RGB)

    screen,gray,img = readScreen()

    # r,g,b,a = screen.split()
    # img= np.array(Image.merge("RGB", (r, g, b)))
    #img = gray

    if process_this_frame:
        faces = detectFaces(gray)
        facenum = len(faces)

    process_this_frame = not process_this_frame

    # for (x,y,w,h) in faces:
    #     numFaces += 1
    #     cropFace,saveFName = saveFaceImg(x,y,w,h,numFaces)
    #
    #     if (os.stat(saveFName).st_size) > 0:
    #         i_w,i_h = Image.open(saveFName).size
    #         if 1.2 >i_w/i_h > 0.8:
    #             sex = estSex(saveFName)
    #             labelFaces(x,y,w,h)
    #         else:
    #             os.remove(saveFName)

    screenshot = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cv2.putText(screenshot, str(facenum), (100, 100), font, 1.0, (255, 0, 0), 1)
    cv2.imshow("screenshot", screenshot)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()