from flask import Flask, render_template, request
import base64
import numpy as np
import cv2
import torch
from model.vit import ViT

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

    class_list = {
        0:"0",
        1:"1",
        2:"2",
        3:"3",
        4:"4",
        5:"5",
        6:"6",
        7:"7",
        8:"8",
        9:"9"
        }

    file = request.form['url']
    file = file[init_Base64:]
    file_decoded = base64.b64decode(file)
    image = np.asarray(bytearray(file_decoded), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    
    #Resizing and reshaping to keep the ratio.
    resized_image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
    vect = np.asarray(resized_image, dtype="uint8")
    vect = vect.reshape(1, 1, 28, 28).astype('float32')
    vect = torch.tensor(vect)

    device = torch.device('cuda')
    weights_path = "model/weights/20230621_134642/best_val_acc0.84.pth"
    vit_model = ViT(image_res=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    vit_model.load_state_dict(torch.load(weights_path, map_location=device))

    out = vit_model(vect)
    prediction = class_list[torch.argmax(out).item()]

    return render_template('vision_transformer/vt_result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)