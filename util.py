import itertools
from predict_gender import *
from faceRecUtil import *

# def detect_face_change(face_num_his,current_face_num):
#     if len(face_num_his) < 5 and face_num_his[-1] > 0:
#         print ("Face detected")
#         return True
#     elif current_face_num != face_num_his[-1]:
#         return False
#     elif collections.deque(itertools.islice(face_num_his, 1, 4)) != collections.deque(itertools.islice(face_num_his, 0, 3)):
#         last9num = collections.deque(itertools.islice(face_num_his, 1, 4))
#         if collections.deque(itertools.islice(last9num, 1, 4)) == collections.deque(itertools.islice(last9num, 0, 3)):
#             print ("Face detected")
#             return True
#         else:
#             return False

def detect_face_change(face_num,last_face_num):
    if face_num != last_face_num:
        return True
    else:
        return False


def saveFaceImg(face_locations,face_name,frame,newpath):
    top, right, bottom, left = face_locations
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    imgDir = newpath+str(face_name)+'/'
    os.mkdir(imgDir)

    margin = int(max(abs(top-bottom),abs(right-left))/2)
    cropFace = frame[top-margin:bottom+margin,left-margin:right+margin]
    saveFName = imgDir+'roi1'+'.png'
    cv2.imwrite(saveFName,cropFace)
    print ('saved face img',face_name)
    return cropFace,saveFName


def saveExistingFaceImg(face_locations,imgDir,frame,name,newpath):

    imgDir = newpath+str(name)+'/'

    countImgs = len(os.listdir(imgDir))

    if countImgs < 5:

        top, right, bottom, left = face_locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        margin = int(max(abs(top-bottom),abs(right-left))/2)
        cropFace = frame[top-margin:bottom+margin,left-margin:right+margin]
        saveFName = imgDir+'roi'+str(countImgs+1)+'.png'
        cv2.imwrite(saveFName,cropFace)

        if (os.stat(saveFName).st_size) > 0:

            i_w,i_h = Image.open(saveFName).size

            if 2.2 >i_w/i_h > 0.4:

                return cropFace,saveFName

        else:

            os.remove(saveFName)

    else:
        return None,None


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

def faceMatched(index_match,know_faces_names,known_face_genders,current_face_genders):
    name = know_faces_names[index_match[0]]
    gender = known_face_genders[index_match[0]]
    current_face_genders.append(gender)

    return name,current_face_genders

def noFaceMatched(text,unknown_face_num,known_faces,know_faces_names,face_encoding):

    print (unknown_face_num)
    unknown_face_num += 1

    name = 'Face'+ str(unknown_face_num)
    print ('adding new face')
    print (unknown_face_num)
    known_faces.append(face_encoding)
    know_faces_names.append(name)

    return name,unknown_face_num,known_faces,know_faces_names

def noGoodFaceSaved(saveFName,unknown_face_num,known_faces,know_faces_names):
    os.remove(saveFName)
    print ('removed face img',saveFName)
    unknown_face_num -= 1
    known_faces.pop()
    know_faces_names.pop()

    return unknown_face_num,know_faces_names,known_faces

def GoodFaceSaved(saveFName,model,known_face_genders,current_face_genders,known_face_genders_mtli,unknown_face_num):


    print ('estiamte gender')
    gender = estSex(saveFName,model)
    print (gender)
    i = unknown_face_num + 1

    known_face_genders.append(gender)
    current_face_genders.append(gender)
    known_face_genders_mtli.append([])
    known_face_genders_mtli[i].append(gender)

    return current_face_genders,known_face_genders,known_face_genders_mtli


def estReadNewImg(saveFName,model,known_face_genders,current_face_genders,unknown_face_num,known_faces,know_faces_names):
    if (os.stat(saveFName).st_size) > 0:

        i_w,i_h = Image.open(saveFName).size

        if 2.2 >i_w/i_h > 0.4:

            current_face_genders,known_face_genders,known_face_gender_mlti = GoodFaceSaved(saveFName,model,known_face_genders,current_face_genders,known_face_genders_mtli,unknown_face_num)
        else:

            unknown_face_num,know_faces_names,known_faces=  noGoodFaceSaved(saveFName,unknown_face_num,known_faces,know_faces_names)
    else:

        unknown_face_num,know_faces_names,known_faces= noGoodFaceSaved(saveFName,unknown_face_num,known_faces,know_faces_names)


    return current_face_genders,known_face_genders,unknown_face_num,know_faces_names,known_faces


def estReadExistImg(saveFName,model,known_face_genders,current_face_genders,name,known_face_genders_mtli):

    print ('estiamte gender')
    est_gender = estSex(saveFName,model)
    print (est_gender)
    i = int(name.split('Face')[1]) + 1

    known_face_genders_mtli[i].append(est_gender)
    gender = collections.Counter(known_face_genders_mtli[i]).most_common(1)[0][0]
    known_face_genders.pop()
    known_face_genders.append(gender)
    print (known_face_genders)
    current_face_genders.pop()
    current_face_genders.append(gender)
    print (current_face_genders)

    return current_face_genders,known_face_genders,known_face_genders_mtli






