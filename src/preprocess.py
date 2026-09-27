import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

DATA_PATH = "dataset/creditcard.csv"
PROCESSED_DIR = "dataset/processed"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("CREDIT CARD FRAUD PREPROCESSING")
print("=" * 50)

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nOriginal dataset:")
print("Features:", X.shape)
print("Normal:", (y == 0).sum())
print("Fraud:", (y == 1).sum())

# Split first to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE ONLY to training data
smote = SMOTE(
    random_state=42,
    sampling_strategy=0.20
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter train/test split:")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nAfter SMOTE:")
print("Training samples:", len(X_train_resampled))
print("Normal:", (y_train_resampled == 0).sum())
print("Fraud:", (y_train_resampled == 1).sum())

# Save processed data
import os
os.makedirs(PROCESSED_DIR, exist_ok=True)

np.save(
    f"{PROCESSED_DIR}/X_train.npy",
    X_train_resampled
)

np.save(
    f"{PROCESSED_DIR}/y_train.npy",
    y_train_resampled.to_numpy()
)

np.save(
    f"{PROCESSED_DIR}/X_test.npy",
    X_test_scaled
)

np.save(
    f"{PROCESSED_DIR}/y_test.npy",
    y_test.to_numpy()
)

joblib.dump(
    scaler,
    f"{PROCESSED_DIR}/scaler.pkl"
)

print("\nPreprocessing completed successfully.")
print("Processed files saved in:", PROCESSED_DIR)
