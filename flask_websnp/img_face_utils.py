import face_recognition
import sys
import time
import os
from PIL import Image, ImageDraw



def makePretty(face_landmarks_list,pil_image,savepath):

    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    pil_image.show()

    pil_image.save(savepath+'pretty.png')

    return pil_image

def detectFaces(face_locations,pil_image):

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

        pil_image.show()

        pil_image.save(newpath+'face.png')

    return pil_image


"""

Create new directories

"""

strings = time.strftime("%Y,%m,%d,%H,%M,%S")

newpath = strings.replace(',','') + '/'

os.makedirs(newpath)

"""
Load image

"""

image = face_recognition.load_image_file(str(sys.argv[1]))

pil_image = Image.fromarray(image)

"""

Make faces prettier

"""


face_landmarks_list = face_recognition.face_landmarks(image)

prettyImg = makePretty(face_landmarks_list,pil_image,newpath)

"""

Detect faces

"""

face_locations = face_recognition.face_locations(image)

faceImg = detectFaces(face_locations,pil_image)