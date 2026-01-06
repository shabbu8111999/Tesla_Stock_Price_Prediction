import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, window_size=60):
    """
    This function creates input-output sequences for time series models
    like RNN and LSTM.
    """
    try:
        # Lists to store input sequences (X) and target values (y)
        X, y = [], []

        # Loop starts from window_size to ensure enough past data
        for i in range(window_size, len(data)):
            # Take previous 'window_size' values as input sequence
            X.append(data[i - window_size:i])

            # Take current value as output
            y.append(data[i])

        # Convert lists into NumPy arrays for model compatibility
        return np.array(X), np.array(y)

    except Exception as e:
        # Print error message if something goes wrong
        print("Error while creating sequences:", e)

        # Return None to avoid program crash
        return None, None


def scale_data(series):
    """
    This function scales the data between 0 and 1 using MinMaxScaler.
    Scaling helps deep learning models converge faster.
    """
    try:
        # Initialize MinMaxScaler with range 0 to 1
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Reshape data to 2D as required by sklearn
        reshaped_series = series.reshape(-1, 1)

        # Fit the scaler and transform the data
        scaled = scaler.fit_transform(reshaped_series)

        # Return scaled data and scaler (needed for inverse transform)
        return scaled, scaler

    except Exception as e:
        # Print error message if scaling fails
        print("Error while scaling data:", e)

        # Return None values to handle failure safely
        return None, None
