import face_recognition
import cv2
import numpy as np
import collections
import itertools
import datetime
import time
from predict_gender import *

strings = time.strftime("%Y,%m,%d,%H,%M,%S")
newpath = strings.replace(',','') + '/'
os.makedirs(newpath)


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
# obama_image = face_recognition.load_image_file("obama1.jpg")
# obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
dingchao_image = face_recognition.load_image_file("obama1.jpg")
dingchao_face_encoding = face_recognition.face_encodings(dingchao_image)[0]
candy_image = face_recognition.load_image_file("biden1.jpg")
candy_face_encoding = face_recognition.face_encodings(candy_image)[0]

known_faces = [
    dingchao_face_encoding,
    candy_face_encoding
]

know_faces_names = ['Dingchao','Candy']

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_genders = ['male','male']
face_landmarks = []
process_this_frame = True
unknown_face_num = 0
face_num_his = collections.deque(maxlen=5)

# Print the location of each facial feature in this image
facial_features = [
    'chin',
    'left_eyebrow',
    'right_eyebrow',
    'nose_bridge',
    'nose_tip',
    'left_eye',
    'right_eye',
    'top_lip',
    'bottom_lip'
]

model = loadVGG()

def detect_face_change(face_num_his,current_face_num):
    if len(face_num_his) < 5 and face_num_his[-1] > 0:
        print ("Face detected")
        return True
    elif current_face_num != face_num_his[-1]:
        return False
    elif collections.deque(itertools.islice(face_num_his, 1, 4)) != collections.deque(itertools.islice(face_num_his, 0, 3)):
        last9num = collections.deque(itertools.islice(face_num_his, 1, 4))
        if collections.deque(itertools.islice(last9num, 1, 4)) == collections.deque(itertools.islice(last9num, 0, 3)):
            print ("Face detected")
            return True
        else:
            return False


def saveFaceImg(face_locations,face_name):
    top, right, bottom, left = face_locations
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    margin = int(max(abs(top-bottom),abs(right-left))/2)
    cropFace = frame[top-margin:bottom+margin,left-margin:right+margin]
    saveFName = newpath+'roi'+str(face_name)+'.png'
    cv2.imwrite(saveFName,cropFace)
    print ('saved face img',face_name)
    return cropFace,saveFName


def estSex(saveFName):
	# resized_img = resize(cropFace)
	# arr = im2Array(resized_img)
	arr = np.array([loadImg2Array(saveFName)])
	sex = predict_vgg(model,arr)
	return sex


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    text = ''

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_num_his.append(len(face_locations))

        face_changed = detect_face_change(face_num_his,len(face_locations))

        if face_changed:
            text += 'face num chagned, doing face match'
            print ('doing match face')
            face_names = []
            for i in range(len(face_locations)):

                face_encoding = face_encodings[i]
                face_location = face_locations[i]

                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces(known_faces, face_encoding)
                print (match)

                if len(np.where(match)[0]) > 0:
                    name = know_faces_names[np.where(match)[0][0]]
                    gender = face_genders[np.where(match)[0][0]]
                else:
                    unknown_face_num += 1
                    name = 'Face'+ str(unknown_face_num)
                    text += 'adding new face'
                    print ('adding new face')
                    known_faces.append(face_encoding)
                    know_faces_names.append(name)

                    cropFace,saveFName = saveFaceImg(face_location,name)

                    if (os.stat(saveFName).st_size) > 0:
                        i_w,i_h = Image.open(saveFName).size
                        if 2.2 >i_w/i_h > 0.4:
                            print ('estiamte gender')
                            gender = estSex(saveFName)
                            print (gender)
                            face_genders.append(gender)



                        else:
                             os.remove(saveFName)
                             print ('removed face img',saveFName)
                             unknown_face_num -= 1


                face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name,gender in zip(face_locations, face_names,face_genders):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name+gender, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, text, (100, 100), font, 1.0, (255, 0, 0), 1)


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()