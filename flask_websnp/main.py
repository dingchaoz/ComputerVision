from flask import Flask, render_template, request, jsonify
from PIL import Image
import face_recognition
from PIL import Image, ImageDraw
import urllib.request
import io

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

    #pil_image.show()

    pil_image.save(savepath+'pretty.png')

    return pil_image



app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/hook', methods=['POST'])
def get_image():
    image_b64 = request.values['imageBase64']

    file = io.BytesIO(urllib.request.urlopen(image_b64).read())
    image_PIL = Image.open(file)
    print ('image opened pil')
    #image_np = np.array(image_PIL)
    #print ('Image received: {}'.format(image_np.shape))
    image_PIL.save('static/img/rec.png')
    print ('Image received')

    image = face_recognition.load_image_file('static/img/rec.png')

    face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)

    prettyImg = makePretty(face_landmarks_list,pil_image,'static/img/')

    return ''




if __name__ == "__main__":
    app.run(host = '0.0.0.0',debug=True)