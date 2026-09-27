import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

TRAIN_X = "dataset/processed/X_train.npy"
TRAIN_Y = "dataset/processed/y_train.npy"
MODEL_PATH = "models/lstm_autoencoder.keras"

X_train = np.load(TRAIN_X)
y_train = np.load(TRAIN_Y)

print("=" * 50)
print("LSTM AUTOENCODER - NORMAL TRANSACTION TRAINING")
print("=" * 50)

# Use only normal transactions for anomaly detection
X_normal = X_train[y_train == 0]

print("Normal training samples:", X_normal.shape[0])
print("Features:", X_normal.shape[1])

# LSTM input:
# samples, timesteps, features
X_normal = X_normal.reshape(
    X_normal.shape[0],
    X_normal.shape[1],
    1
)

timesteps = X_normal.shape[1]

inputs = Input(shape=(timesteps, 1))

encoded = LSTM(
    32,
    activation="tanh",
    return_sequences=False
)(inputs)

decoded = RepeatVector(timesteps)(encoded)

decoded = LSTM(
    32,
    activation="tanh",
    return_sequences=True
)(decoded)

outputs = TimeDistributed(
    Dense(1)
)(decoded)

model = Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)

history = model.fit(
    X_normal,
    X_normal,
    epochs=15,
    batch_size=256,
    validation_split=0.10,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=1
)

print()
print("Training completed successfully.")
print("Model saved to:", MODEL_PATH)
