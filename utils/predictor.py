import numpy as np


def predict_future(model, last_sequence, days, scaler):
    """
    This function predicts future stock prices using a trained
    RNN/LSTM model based on the last available sequence.
    """
    try:
        # List to store future predictions (scaled values)
        predictions = []

        # Ensure the input sequence has correct shape (60, 1)
        # 60 = time steps, 1 = feature (Close price)
        current_seq = last_sequence.reshape(60, 1)

        # Loop for the number of future days to predict
        for _ in range(days):
            # Reshape input to 3D format required by RNN/LSTM:
            # (batch_size=1, time_steps=60, features=1)
            pred = model.predict(
                current_seq.reshape(1, 60, 1),
                verbose=0
            )

            # Store the predicted value
            predictions.append(pred[0][0])

            # Update the sequence:
            # Remove the oldest value and append the new prediction
            current_seq = np.vstack([
                current_seq[1:],          # drop first value
                [[pred[0][0]]]             # add new prediction
            ])

        # Convert predictions list to NumPy array
        predictions = np.array(predictions).reshape(-1, 1)

        # Convert scaled predictions back to original price scale
        predictions = scaler.inverse_transform(predictions)

        # Return final predicted prices
        return predictions

    except Exception as e:
        # Print error message if prediction fails
        print("Error during future prediction:", e)

        # Return None to safely handle failure
        return None
