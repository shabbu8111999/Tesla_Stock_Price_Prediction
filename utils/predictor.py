# Import NumPy for numerical operations
import numpy as np


# Function to predict future stock prices
def predict_future(model, last_sequence, days, scaler):
    """
    This function predicts future stock prices using a trained
    RNN or LSTM model based on the last available sequence.
    """

    try:
        # Create an empty list to store predicted values
        predictions = []

        # Reshape the last known sequence into (60, 1)
        current_seq = last_sequence.reshape(60, 1)

        # Loop to predict prices for the given number of future days
        for _ in range(days):

            # Reshape input into 3D format required by RNN/LSTM
            pred = model.predict(
                current_seq.reshape(1, 60, 1),
                verbose=0  # Suppress prediction logs
            )

            # Extract the predicted value and store it
            predictions.append(pred[0][0])

            # Update the sequence for the next prediction
            current_seq = np.vstack([
                current_seq[1:],       # drop the oldest value
                [[pred[0][0]]]          # add the new prediction
            ])

        # Convert predictions list into NumPy array
        predictions = np.array(predictions).reshape(-1, 1)

        # Convert scaled predictions back to original stock price values
        predictions = scaler.inverse_transform(predictions)

        # Return the final predicted prices
        return predictions

    except Exception as e:
        # Print error message if prediction fails
        print("Error during future prediction:", e)

        # Return None to safely handle any failure
        return None



# Function to add confidence range to predictions
def add_confidence_range(predictions, percentage=0.02):
    """
    Adds an approximate confidence range to predictions.

    percentage = 0.02 means ±2% range
    """

    # Calculate lower bound (prediction - percentage)
    lower = predictions * (1 - percentage)

    # Calculate upper bound (prediction + percentage)
    upper = predictions * (1 + percentage)

    # Return lower and upper confidence values
    return lower, upper
