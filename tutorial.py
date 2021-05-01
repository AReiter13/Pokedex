from flask import Flask, redirect, url_for, render_template
from flask_ngrok import run_with_ngrok

app = Flask(__name__)

run_with_ngrok(app)

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
