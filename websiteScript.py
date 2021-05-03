from flask import Flask, redirect, url_for, render_template, request
from flask_ngrok import run_with_ngrok
import requests, re, time
import torch, torchvision
from torch import nn, optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
import numpy
import cv2

device = torch.device('cuda:0') 
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 75)
model.to(device)
model.load_state_dict(torch.load('websitestuff/model'))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

ALLOWED_FILES = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILES

def makePrediction(image):
  xform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), torchvision.transforms.RandomHorizontalFlip(p=0.5)])
  image = xform(image).to(device)
  model.eval()
  image = image.unsqueeze(0)
  output = model(image)
  _, pred = torch.max(output.detach(), 1)
  return pred

app = Flask(__name__)

run_with_ngrok(app)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
      if 'im' not in request.files:
            return render_template("Home.html")
      img = request.files['im']
      if img.filename == '':
            return render_template("Home.html")
      if img and allowed_file(img.filename):
            pred = makePrediction(Image.open(img)).item()+1
            if pred == 58:
                return redirect(url_for("growlithe"))
            else:
                return redirect(url_for("rattata"))
      else:
            return render_template("Home.html")
    else:
      return render_template("Home.html")
    

@app.route("/growlithe/")
def growlithe():
    return render_template("Growlithe.html")

@app.route("/rattata/")
def rattata():
    return render_template("Rattata.html")

@app.route("/ppp/")
def ppp():
    return render_template("pokemon.html")


if __name__ == "__main__": 
    app.run()
