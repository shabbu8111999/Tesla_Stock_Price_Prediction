# Import standard libraries
import os
from datetime import timedelta

# Import data handling libraries
import pandas as pd
import numpy as np

# Import Flask components
from flask import Flask, render_template, request

# Import model loading function
from tensorflow.keras.models import load_model

# Configure matplotlib to work with Flask (no GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import helper functions for preprocessing and prediction
from utils.preprocess import scale_data
from utils.predictor import predict_future, add_confidence_range


# Create Flask application instance
app = Flask(__name__)


# Load and prepare dataset once when app starts
try:
    # Read Tesla stock CSV file and parse Date column
    df = pd.read_csv("data/TSLA.csv", parse_dates=["Date"])

    # Extract closing prices
    close_price = df["Close"].values

    # Scale closing prices and get scaler
    scaled, scaler = scale_data(close_price)

    # Store last 60 days sequence for future prediction
    last_sequence = scaled[-60:].reshape(60, 1)

except Exception as e:
    # Handle errors during data loading or preprocessing
    print("Error while loading or preprocessing data:", e)
    df, scaler, last_sequence = None, None, None


# Load trained RNN and LSTM models
try:
    rnn = load_model("model/rnn_model.h5", compile=False)
    lstm = load_model("model/lstm_model.h5", compile=False)

except Exception as e:
    # Handle errors if models are not loaded
    print("Error while loading models:", e)
    rnn, lstm = None, None


# Define main route for the application
@app.route("/", methods=["GET", "POST"])
def index():

    # Initialize variables for results and errors
    results = None
    plot_url = None
    error_message = None

    try:
        # Handle form submission
        if request.method == "POST":

            # Get number of days and selected model from form
            days = int(request.form["days"])
            model_type = request.form["model"]

            # Validate input range
            if days <= 0 or days > 60:
                error_message = "Please enter a number of days between 1 and 60."
                raise ValueError("Invalid days range")

            # Select model based on user choice
            model = lstm if model_type == "lstm" else rnn

            # Check if model is loaded
            if model is None:
                error_message = "Model is not available."
                raise ValueError("Model not loaded")

            # Predict future stock prices
            prediction = predict_future(model, last_sequence, days, scaler)

            # Check if prediction was successful
            if prediction is None:
                error_message = "Prediction failed. Please try again."
                raise ValueError("Prediction returned None")

            # Calculate confidence range
            lower, upper = add_confidence_range(prediction)

            # Generate future dates starting from last dataset date
            last_date = df["Date"].iloc[-1]
            future_dates = [
                last_date + timedelta(days=i + 1)
                for i in range(days)
            ]

            # Prepare results for UI display
            results = []
            for i in range(days):
                results.append({
                    "date": future_dates[i].date(),
                    "price": round(float(prediction[i][0]), 2),
                    "low": round(float(lower[i][0]), 2),
                    "high": round(float(upper[i][0]), 2),
                })

            # Define plot file name and path
            plot_url = "prediction_plot.png"
            plot_path = os.path.join("static", plot_url)

            # Create prediction plot
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
        # Catch and print handled errors
        print("Handled error:", e)

    # Render HTML template with prediction data
    return render_template(
        "index.html",
        prediction=results,
        plot_url=plot_url,
        error_message=error_message
    )


# Run Flask app in debug mode
if __name__ == "__main__":
    app.run(debug=True)
