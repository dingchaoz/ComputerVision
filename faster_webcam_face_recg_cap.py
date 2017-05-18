import time
from util import *


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

    #print (unknown_face_num)
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
        screen_face_locations = []
        #for face_loc in face_locations:
            #screen_face_locations.append(face_loc)


        #print (face_locations,face_encodings)

        #face_changed = detect_face_change(len(face_locations),last_face_num)
        face_changed = True
        if face_changed or get_moreface:

            #print ('doing match face')
            face_names = []
            current_face_genders = []
            screen_face_locations = []
            for i in range(len(face_locations)):

                face_encoding = face_encodings[i]
                face_location = face_locations[i]

                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces(known_faces, face_encoding,tolerance = 0.6)

                indices_match = np.where(match)[0]

                if len([x for x in indices_match if x <=3]) > 1:
                    index_match = [np.argmin(face_recognition.face_distance(known_faces, face_encoding)[:3])]
                else:
                    index_match = indices_match

                #print (index_match)

                if len(index_match) > 0:

                    name,current_face_genders = faceMatched(index_match,know_faces_names,known_face_genders,current_face_genders)

                    imgDir = newpath+str(name)+'/'

                    cropFace,saveFName,get_moreface = saveExistingFaceImg(face_location,imgDir,frame,name,newpath,get_moreface)

                    if saveFName != None:
                        current_face_genders,known_face_genders,known_face_genders_mtli = estReadExistImg(saveFName,model,known_face_genders,current_face_genders,name,known_face_genders_mtli)

                else:

                    name,unknown_face_num,known_faces,know_faces_names,get_moreface = noFaceMatched(text,unknown_face_num,known_faces,know_faces_names,face_encoding,get_moreface)

                    cropFace,saveFName = saveFaceImg(face_location,name,frame,newpath)

                    current_face_genders,known_face_genders,unknown_face_num,know_faces_names,known_faces = estReadNewImg(saveFName,model,known_face_genders,current_face_genders,unknown_face_num,known_faces,know_faces_names)

                face_names.append(name)
                screen_face_locations.append(face_location)
                #print(face_location,name)
                #print (time.time() - start)

    last_face_num = len(face_locations)

    process_this_frame = not process_this_frame

    # Display result and label faces
    displayRes(screen_face_locations, face_names,current_face_genders,frame,text)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()