import face_recognition



# Load a sample picture and learn how to recognize it.

kface1_image = face_recognition.load_image_file("stiles.jpg")
kface1_face_encoding = face_recognition.face_encodings(kface1_image)[0]
kface2_image = face_recognition.load_image_file("biden1.jpg")
kface2_face_encoding = face_recognition.face_encodings(kface2_image)[0]
kface3_image = face_recognition.load_image_file("profile.jpg")
kface3_face_encoding = face_recognition.face_encodings(kface3_image)[0]

known_faces = [
    kface1_face_encoding,
    kface2_face_encoding,
    kface3_face_encoding
]

know_faces_names = ['Mike','Biden','Dingchao']

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_genders = ['?','male','male']
known_face_genders_mtli = [[],[],[]]
current_face_genders = []
face_landmarks = []
process_this_frame = True
unknown_face_num = 0
last_face_num = 0
get_moreface = False

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