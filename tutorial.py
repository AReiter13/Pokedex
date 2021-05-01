from flask import Flask, redirect, url_for, render_template
import requests, re, time
from torch import nn, optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from google.colab import files

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/predict/")
def predict():
    
    return f"predicting"

@app.route("/growlithe/")
def growlithe():
    return render_template("growlithe.html")

@app.route("/notGrowlithe")
def notGrowlithe():
    return f"not Growlithe"

if __name__ == "__main__": 
    app.run()
