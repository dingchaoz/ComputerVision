import sys
import time

# Load the pre-trained face and eye classifier xml file, which are stored in opencv/data/haarcascades/folder

"""
Usage: python Face_detection.py trump_melania.jpg
"""

from util import *
import face_recognition

from PIL import Image, ImageDraw


if __name__ == "__main__":
    #main(sys.argv[1:])



    strings = time.strftime("%Y,%m,%d,%H,%M,%S")
    newpath = strings.replace(',','') + '/'
    os.makedirs(newpath)

    model = loadVGG()

    frame = face_recognition.load_image_file(str(sys.argv[1]))

    text = ''


    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    screen_face_locations = []
            #for face_loc in face_locations:
                #screen_face_locations.append(face_loc)


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

        name,unknown_face_num,known_faces,know_faces_names,get_moreface = noFaceMatched(text,unknown_face_num,known_faces,know_faces_names,face_encoding,get_moreface)

        cropFace,saveFName = saveFaceImg(face_location,name,frame,newpath)

        current_face_genders,known_face_genders,unknown_face_num,know_faces_names,known_faces = estReadNewImg(saveFName,model,known_face_genders,current_face_genders,unknown_face_num,known_faces,know_faces_names)

        face_names.append(name)
        screen_face_locations.append(face_location)
        #print(face_location,name)
        #print (time.time() - start)

    last_face_num = len(face_locations)

    # Display result and label faces
    displayRes(screen_face_locations, face_names,current_face_genders,frame,text)


    # # show image with rectangular
    # cv2.imshow('img',frame)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()







