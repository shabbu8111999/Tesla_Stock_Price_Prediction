# Import NumPy for numerical operations
import numpy as np

# Import MinMaxScaler to scale values between 0 and 1
from sklearn.preprocessing import MinMaxScaler


# Function to create input-output sequences for time series
def create_sequences(data, window_size=60):
    """
    This function prepares data for RNN/LSTM models.
    It converts a single time series into input (X) and output (y) pairs.
    """

    try:
        # Create empty lists to store inputs (X) and outputs (y)
        X = []
        y = []

        # Start loop from window_size so we have enough past data
        for i in range(window_size, len(data)):

            # Take the previous 'window_size' values as input
            X.append(data[i - window_size:i])

            # Take the current value as the target output
            y.append(data[i])

        # Convert lists into NumPy arrays (required by deep learning models)
        X = np.array(X)
        y = np.array(y)

        # Return input sequences and corresponding outputs
        return X, y

    except Exception as e:
        # Print error message if sequence creation fails
        print("Error while creating sequences:", e)

        # Return None values to prevent program crash
        return None, None


# Function to scale data using Min-Max scaling
def scale_data(series):
    """
    This function scales stock price data between 0 and 1.
    Scaling improves training stability and model performance.
    """

    try:
        # Create a MinMaxScaler object with range [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Reshape data into 2D format
        reshaped_series = series.reshape(-1, 1)

        # Fit the scaler on data and transform it
        scaled_data = scaler.fit_transform(reshaped_series)

        # Return scaled data and scaler
        # Scaler is needed later to convert predictions back
        return scaled_data, scaler

    except Exception as e:
        # Print error message if scaling fails
        print("Error while scaling data:", e)

        # Return None values to handle failure safely
        return None, None
