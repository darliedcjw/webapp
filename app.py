from flask import Flask, render_template, request, redirect
import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from ResNet152.model import ResNet152
from werkzeug.utils import secure_filename


init_Base64=21


app = Flask(__name__)
app.config['upload_folder'] = 'upload'

@app.route("/")
def home():
    
    main_description = "Welcome to Darryl's BrainSpace"
    
    sub_description = "This page contains some of my past projects."
    
    projects=["MNIST Prediction"]

    return render_template('layout.html',
                           main_description=main_description,
                           sub_description=sub_description,
                           projects=projects)

@app.route("/mnist")
def mnist():
    return render_template('mnist/mnist_home.html')

@app.route("/mnist_predict")
def mnist_predict():

    device = torch.device('cuda')
    
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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    ds = MNIST(root="ResNet152/datasets", train=False, transform=transform)
    rand = torch.randint(low=0, high=100, size=(1,)).item()
    image = ds[rand][0]
    label = ds[rand][1]

    if not os.path.exists('static/examples'):
        os.makedirs('static/examples')
    
    save_path = os.path.join('static/examples', '{}.jpeg'.format(rand))
    image_save = image.numpy().transpose(1, 2, 0) * 255
    image_save = cv2.resize(image_save, [128, 128])
    print(image_save.shape)
    cv2.imwrite(save_path, image_save)

    image = image.unsqueeze(dim=0).repeat(1, 3, 1, 1).to(device)

    weights_path = "ResNet152/logs/mnist/checkpoint_best_0.7509_0.9510.pth"
    checkpoint = torch.load(weights_path, map_location=device)

    resnet152 = ResNet152(in_channels=3, num_classes=10).to(device)
    resnet152.load_state_dict(checkpoint['model'])
    
    resnet152.eval()
    # vit_model = ViT(image_res=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    # vit_model.load_state_dict(torch.load(weights_path, map_location=device))

    out = resnet152(image)
    prediction = class_list[torch.argmax(out).item()]

    return render_template('mnist/resnet_result.html', prediction=prediction, label=label, save_path=save_path)


if __name__ == "__main__":
    app.run(debug=True)