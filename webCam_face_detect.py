# Reference
# https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/
import cv2
import sys
import datetime

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
margin = 150

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #http://stackoverflow.com/questions/31932588/opencv-face-recognition-get-coordinates-of-bounding-box-around-image
        cropFace = frame[y-margin:y+h+margin,x-margin:x+w+margin]
    	saveFName = 'roi'+str(datetime.datetime.now())+'.png'
    	cv2.imwrite(saveFName,cropFace)


        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
    	cv2.putText(frame,' Male,28',(x+w, y+h), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    	

    # Display the resulting frame
    cv2.imshow('Video', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
