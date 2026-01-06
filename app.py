import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from datetime import timedelta
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use("Agg")   # IMPORTANT for Flask
import matplotlib.pyplot as plt

from utils.preprocess import scale_data
from utils.predictor import predict_future, add_confidence_range


# -------------------------------------------------
# Flask App Initialization
# -------------------------------------------------
app = Flask(__name__)


# -------------------------------------------------
# Load and Prepare Dataset (Executed Once)
# -------------------------------------------------
try:
    df = pd.read_csv("data/TSLA.csv", parse_dates=["Date"])

    close_price = df["Close"].values
    scaled, scaler = scale_data(close_price)

    last_sequence = scaled[-60:].reshape(60, 1)

except Exception as e:
    print("Error while loading or preprocessing data:", e)
    df, scaler, last_sequence = None, None, None


# -------------------------------------------------
# Load Trained Models
# -------------------------------------------------
try:
    rnn = load_model("model/rnn_model.h5", compile=False)
    lstm = load_model("model/lstm_model.h5", compile=False)

except Exception as e:
    print("Error while loading models:", e)
    rnn, lstm = None, None


# -------------------------------------------------
# Main Route
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    results = None
    plot_url = None
    error_message = None

    try:
        if request.method == "POST":

            days = int(request.form["days"])
            model_type = request.form["model"]

            if days <= 0 or days > 60:
                error_message = "Please enter a number of days between 1 and 60."
                raise ValueError("Invalid days range")


            model = lstm if model_type == "lstm" else rnn
            if model is None:
                error_message = "Model is not available."
                raise ValueError("Model not loaded")

            prediction = predict_future(model, last_sequence, days, scaler)
            if prediction is None:
                error_message = "Prediction failed. Please try again."
                raise ValueError("Prediction returned None")

            lower, upper = add_confidence_range(prediction)

            last_date = df["Date"].iloc[-1]
            future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]

            results = []
            for i in range(days):
                results.append({
                    "date": future_dates[i].date(),
                    "price": round(float(prediction[i][0]), 2),
                    "low": round(float(lower[i][0]), 2),
                    "high": round(float(upper[i][0]), 2),
                })

            plot_url = "prediction_plot.png"
            plot_path = os.path.join("static", plot_url)

            plt.figure(figsize=(8, 4))
            plt.plot(future_dates, prediction, marker="o", label="Prediction")
            plt.fill_between(
                future_dates,
                lower.flatten(),
                upper.flatten(),
                alpha=0.2,
                label="Confidence Range"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

    except Exception as e:
        print("Handled error:", e)

    return render_template(
        "index.html",
        prediction=results,
        plot_url=plot_url,
        error_message=error_message
    )


if __name__ == "__main__":
    app.run(debug=True)
