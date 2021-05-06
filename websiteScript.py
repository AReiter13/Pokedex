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
            #REDIRECTIONS
            if pred == 38:
                return redirect(url_for("ninetales"))
            elif pred == 39:
                return redirect(url_for("jigglypuff"))
            elif pred == 40:
                return redirect(url_for("wigglytuff"))
            elif pred == 41:
                return redirect(url_for("zubat"))
            elif pred == 42:
                return redirect(url_for("golbat"))
            elif pred == 43:
                return redirect(url_for("oddish"))
            elif pred == 44:
                return redirect(url_for("gloom"))
            elif pred == 45:
                return redirect(url_for("vileplume"))
            elif pred == 46:
                return redirect(url_for("paras"))
            elif pred == 47:
                return redirect(url_for("parasect"))
            elif pred == 48:
                return redirect(url_for("venonat"))                                                                                
            elif pred == 49:
                return redirect(url_for("venomoth"))
            elif pred == 50:
                return redirect(url_for("diglett"))
            elif pred == 51:
                return redirect(url_for("dugtrio"))
            elif pred == 52:
                return redirect(url_for("meowth"))
            elif pred == 53:
                return redirect(url_for("persian"))
            elif pred == 54:
                return redirect(url_for("psyduck"))
            elif pred == 55:
                return redirect(url_for("golduck"))
            elif pred == 56:
                return redirect(url_for("mankey"))    
            else:
                return redirect(url_for("rattata"))
      else:
            return render_template("Home.html")
    else:
      return render_template("Home.html")
    

@app.route("/ninetales/")
def ninetales():
    return render_template("Ninetales.html")

@app.route("/jigglypuff/")
def jigglypuff():
    return render_template("Jigglypuff.html")

@app.route("/wigglytuff/")
def wigglytuff():
    return render_template("Wigglytuff.html")

@app.route("/zubat/")
def zubat():
    return render_template("Zubat.html")

@app.route("/golbat/")
def golbat():
    return render_template("Golbat.html")

@app.route("/oddish/")
def oddish():
    return render_template("Oddish.html")

@app.route("/gloom/")
def gloom():
    return render_template("Gloom.html")    

@app.route("/vileplume/")
def vileplume():
    return render_template("Vileplume.html")

@app.route("/paras/")
def paras():
    return render_template("Paras.html")

@app.route("/parasect/")
def parasect():
    return render_template("Parasect.html")

@app.route("/venonat/")
def venonat():
    return render_template("Venonat.html")

@app.route("/venomoth/")
def venomoth():
    return render_template("Venomoth.html")

@app.route("/diglett/")
def diglett():
    return render_template("Diglett.html")

@app.route("/dugtrio/")
def dugtrio():
    return render_template("Dugtrio.html")

@app.route("/meowth/")
def meowth():
    return render_template("Meowth.html")

@app.route("/persian/")
def persian():
    return render_template("Persian.html")

@app.route("/psyduck/")
def psyduck():
    return render_template("Psyduck.html")

@app.route("/golduck/")
def golduck():
    return render_template("Golduck.html")

@app.route("/mankey/")
def mankey():
    return render_template("Mankey.html")

@app.route("/rattata/")
def rattata():
    return render_template("Rattata.html")



if __name__ == "__main__": 
    app.run()
