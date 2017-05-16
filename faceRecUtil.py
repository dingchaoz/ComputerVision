import face_recognition



# Load a sample picture and learn how to recognize it.

dingchao_image = face_recognition.load_image_file("obama1.jpg")
dingchao_face_encoding = face_recognition.face_encodings(dingchao_image)[0]
candy_image = face_recognition.load_image_file("biden1.jpg")
candy_face_encoding = face_recognition.face_encodings(candy_image)[0]

known_faces = [
    dingchao_face_encoding,
    candy_face_encoding
]

know_faces_names = ['Dingchao','Biden']

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_genders = ['male','male']
known_face_genders_mtli = [[],[]]
current_face_genders = []
face_landmarks = []
process_this_frame = True
unknown_face_num = 0
last_face_num = 0
get_moreface = True

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