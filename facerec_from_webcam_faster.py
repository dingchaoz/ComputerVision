import time
from util import *
from faceRecUtil import *

start = time.time()

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

# Load model
model = loadVGG()


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #small_frame = frame

    text = ''

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_num_his.append(len(face_locations))

        #face_changed = detect_face_change(face_num_his,len(face_locations))

        #if face_changed:
        text += 'face num chagned, doing face match'
        #print ('doing match face')
        face_names = []
        current_face_genders = []
        for i in range(len(face_locations)):

            face_encoding = face_encodings[i]
            face_location = face_locations[i]

            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding,tolerance = 0.6)
            print (match)

            if len(np.where(match)[0]) > 0:
                name = know_faces_names[np.where(match)[0][0]]
                gender = known_face_genders[np.where(match)[0][0]]
                current_face_genders.append(gender)
            else:
                unknown_face_num += 1
                name = 'Face'+ str(unknown_face_num)
                text += 'adding new face'
                print ('adding new face')
                known_faces.append(face_encoding)
                know_faces_names.append(name)


                cropFace,saveFName = saveFaceImg(face_location,name,frame,newpath)

                if (os.stat(saveFName).st_size) > 0:
                    i_w,i_h = Image.open(saveFName).size
                    if 2.2 >i_w/i_h > 0.4:
                        print ('estiamte gender')
                        gender = estSex(saveFName,model)
                        print (gender)
                        known_face_genders.append(gender)
                        current_face_genders.append(gender)


                    else:
                        os.remove(saveFName)
                        print ('removed face img',saveFName)
                        unknown_face_num -= 1
                        known_faces.pop()
                        know_faces_names.pop()

                else:
                     os.remove(saveFName)
                     print ('removed face img',saveFName)
                     unknown_face_num -= 1
                     known_faces.pop()
                     know_faces_names.pop()


            face_names.append(name)
            print (time.time() - start)

    process_this_frame = not process_this_frame

    # Display result and label faces
    displayRes(face_locations, face_names,current_face_genders,frame,text)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()