from flask import Flask, render_template, request, jsonify
from PIL import Image
import urllib.request
import io



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
    image_PIL.save('rec.png')
    print ('Image received')
    return ''


if __name__ == "__main__":
    app.run(debug=True)