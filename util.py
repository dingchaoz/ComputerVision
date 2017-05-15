import itertools
from predict_gender import *

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


def saveFaceImg(face_locations,face_name,frame,newpath):
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


def estSex(saveFName,model):
	# resized_img = resize(cropFace)
	# arr = im2Array(resized_img)
	arr = np.array([loadImg2Array(saveFName)])
	sex = predict_vgg(model,arr)
	return sex


def displayRes(face_locations, face_names,face_genders,frame,text):

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
        cv2.putText(frame, name+' ' +gender, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, text+str(len(face_locations)), (100, 100), font, 1.0, (255, 0, 0), 1)


    # Display the resulting image
    cv2.imshow('Video', frame)
