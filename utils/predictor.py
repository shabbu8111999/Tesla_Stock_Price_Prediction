import numpy as np

def predict_future(model, last_sequence, days, scaler):
    predictions = []

    # Ensure correct shape: (60, 1)
    current_seq = last_sequence.reshape(60, 1)

    for _ in range(days):
        # Model expects (1, 60, 1)
        pred = model.predict(
            current_seq.reshape(1, 60, 1),
            verbose=0
        )

        predictions.append(pred[0][0])

        # Roll the sequence
        current_seq = np.vstack([
            current_seq[1:],
            [[pred[0][0]]]
        ])

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions
