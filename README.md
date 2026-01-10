# 📈 Tesla Stock Price Prediction using RNN & LSTM
## 📌 Project Overview

This project predicts future Tesla stock prices using Deep Learning models — Simple RNN and LSTM.

A user-friendly Flask web application is built where users can select a model, choose the number of future days, and view:

- Predicted stock prices
- Confidence range (upper & lower bounds)
- A prediction graph
- Model comparison (RNN vs LSTM)

This project is designed in a beginner-friendly way and focuses on understanding the end-to-end flow of a Machine Learning + Web application.

---

## 🎯 Project Objectives

- Learn how time-series stock data works
- Understand RNN and LSTM models practically
- Build a simple Flask-based ML web app
- Visualize predictions with confidence ranges
- Create a clean UI for demonstration and interviews

---

## 🛠️ Technologies Used

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- HTML & CSS

---

## Project Structure

<pre>
Tesla-Stock-Prediction/
│
├── app.py                     # Main Flask application
├── README.md                  # Project documentation
│
├── data/
│   └── TSLA.csv               # Tesla stock dataset
│
├── model/
│   ├── rnn_model.h5           # Trained RNN model
│   └── lstm_model.h5          # Trained LSTM model
│
├── templates/
│   └── index.html             # Frontend HTML file
│
├── static/
│   ├── style.css              # CSS styling
│   └── prediction_plot.png    # Generated prediction plot
│
├── utils/
│   ├── preprocess.py          # Data scaling and sequence creation
│   └── predictor.py           # Future prediction logic
│
└── requirements.txt           # Required Python packages
</pre>

---

## 📊 Dataset Information

- Dataset contains historical Tesla stock prices
- Columns include: Date, Open, High, Low, Close, Volume
- Only the Close price is used for prediction
- Data is scaled using MinMaxScaler (0 to 1)

---

## 🧠 Models Used
🔹 Simple RNN

- Captures short-term patterns
- Simple architecture
- Less effective for long-term dependencies

🔹 LSTM

- Handles long-term memory better
- Performs better for stock price prediction
- More stable predictions

---

## 🔁 How Prediction Works (Simple Flow)

- Load historical Tesla stock data
- Scale closing prices between 0 and 1
- Use the last 60 days as input
- Predict one future day at a time
- Append prediction back into input sequence
- Repeat for selected number of days
- Convert predictions back to real prices
- Add confidence range (±2%)

---

## 🖥️ Web Application Features

- Model selection (RNN / LSTM)
- Input number of future days (1–60)
- Loading spinner during prediction
- Prediction table with confidence range
- Prediction graph visualization
- Model accuracy comparison bars
- Clean dark finance-style UI

---

## ▶️ How to Run the Project
### Clone The Repository First
```bash
git clone https://github.com/your-username/tesla-stock-prediction.git (Add your GitHub Project Link)
cd tesla-stock-prediction (Add your own path here)
```

### Install Required Packages with the Help of UV
```bash
uv add (your installation requirements)
```

### Run the Flask
```bash
uv run app.py
```

### Open in Browser
```bash
http://127.0.0.1:5000/
```

---

## 🧪 Input Validation

- Days must be between 1 and 60
- Proper error messages shown for invalid input
- Safe handling of prediction failures

---

## 📈 Output Example

- Date-wise predicted stock prices
- Lower and upper confidence bounds
- Line plot with shaded confidence area

---

## ⚠️ Disclaimer

***This project is for learning and educational purposes only.***
***Stock market predictions are uncertain and should not be used for real financial decisions.***

---

## 🙌 What I Learned from This Project

- Time-series data preprocessing
- RNN vs LSTM differences
- Model deployment using Flask
- Connecting ML models with frontend
- Error handling and user experience
- End-to-end ML project workflow

---

## 🚀 Future Improvements

- Real-time stock data API
- Dynamic confidence calculation
- Accuracy fetched from backend
- Mobile responsive UI
- Deployment on cloud (Heroku / Render)

---

## 👤 Author

**Shabareesh Nair**
**AI / ML Enthusiast**
**Deep Learning Projects**