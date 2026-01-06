from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import scale_data
from utils.predictor import predict_future

app = Flask(__name__)

df = pd.read_csv("data/TSLA.csv")
close_price = df["Close"].values
scaled, scaler = scale_data(close_price)

last_sequence = scaled[-60:].reshape(60, 1)

rnn = load_model("model/rnn_model.h5", compile=False)
lstm = load_model("model/lstm_model.h5", compile=False)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        days = int(request.form["days"])
        model_type = request.form["model"]

        model = lstm if model_type == "lstm" else rnn
        prediction = predict_future(model, last_sequence, days, scaler)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
