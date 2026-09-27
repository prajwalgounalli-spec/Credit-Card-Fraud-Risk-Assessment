import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("models/lstm_autoencoder.keras")
scaler = joblib.load("dataset/processed/scaler.pkl")

print("=" * 50)
print("CREDIT CARD FRAUD RISK PREDICTION")
print("=" * 50)
print()
print("Enter 30 values in ONE LINE:")
print("Time V1 V2 ... V28 Amount")
print()

values = input("Transaction: ").split()

if len(values) != 30:
    print("ERROR: Please enter exactly 30 values.")
    print("You entered:", len(values))
    exit()

values = [float(value) for value in values]

transaction = np.array(values).reshape(1, -1)
transaction_scaled = scaler.transform(transaction)

transaction_lstm = transaction_scaled.reshape(1, 30, 1)

reconstructed = model.predict(transaction_lstm, verbose=0)

anomaly_score = float(
    np.mean(np.square(transaction_lstm - reconstructed))
)

errors = np.load("results/reconstruction_errors.npy")
labels = np.load("results/test_labels.npy")

normal_errors = errors[labels == 0]

low_threshold = np.percentile(normal_errors, 95)
high_threshold = np.percentile(normal_errors, 99.5)

if anomaly_score <= low_threshold:
    prediction = "NORMAL"
    risk = "LOW"
elif anomaly_score <= high_threshold:
    prediction = "SUSPICIOUS"
    risk = "MEDIUM"
else:
    prediction = "POTENTIAL FRAUD"
    risk = "HIGH"

print()
print("=" * 50)
print("PREDICTION RESULT")
print("=" * 50)
print(f"Anomaly Score : {anomaly_score:.4f}")
print(f"Prediction    : {prediction}")
print(f"Risk Level    : {risk}")
print("=" * 50)
