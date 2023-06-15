from flask import Flask, render_template, request
import base64
import numpy as np
import cv2

init_Base64=21

app = Flask(__name__)

@app.route("/")
def home():
    
    main_description = "Welcome to Darryl's BrainSpace"
    
    sub_description = "This page contains some of my past projects."
    
    projects=["Vision Transformer"]

    return render_template('layout.html',
                           main_description=main_description,
                           sub_description=sub_description,
                           projects=projects)

@app.route("/vision_transformer")
def vision_transformer():
    return render_template('vision_transformer/vt_home.html')

@app.route("/predict", methods=["POST"])
def predict():
    file = request.form['url']
    file = file[init_Base64:]
    file_decoded = base64.b64decode(file)
    image = np.asarray(bytearray(file_decoded), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    
    #Resizing and reshaping to keep the ratio.
    resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
    vect = np.asarray(resized, dtype="uint8")
    vect = vect.reshape(1, 1, 28, 28).astype('float32')
    
    

    return render_template('vision_transformer/vt_result.html', prediction='LOL')


if __name__ == "__main__":
    app.run(debug=True)