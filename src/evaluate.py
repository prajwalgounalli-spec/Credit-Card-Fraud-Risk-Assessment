import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

TEST_X = "dataset/processed/X_test.npy"
TEST_Y = "dataset/processed/y_test.npy"
MODEL_PATH = "models/lstm_autoencoder.keras"

X_test = np.load(TEST_X)
y_test = np.load(TEST_Y)

model = tf.keras.models.load_model(MODEL_PATH)

print("=" * 50)
print("LSTM AUTOENCODER FRAUD EVALUATION")
print("=" * 50)

# Reshape for LSTM
X_test_lstm = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("Test samples:", X_test.shape[0])

# Reconstruct transactions
reconstructed = model.predict(
    X_test_lstm,
    batch_size=256,
    verbose=1
)

# Reconstruction error
errors = np.mean(
    np.square(X_test_lstm - reconstructed),
    axis=(1, 2)
)

# Use normal transactions to determine threshold
normal_errors = errors[y_test == 0]

threshold = np.percentile(
    normal_errors,
    99.5
)

print()
print("Anomaly Threshold:", threshold)

# Predict anomaly/fraud
y_pred = (errors > threshold).astype(int)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    errors
)

print()
print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title("LSTM Autoencoder Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(
    y_test,
    errors
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LSTM Autoencoder ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300
)

plt.close()

# Save reconstruction errors
np.save(
    "results/reconstruction_errors.npy",
    errors
)

np.save(
    "results/predictions.npy",
    y_pred
)

np.save(
    "results/test_labels.npy",
    y_test
)

with open(
    "results/threshold.txt",
    "w"
) as file:
    file.write(str(threshold))

print()
print("Evaluation completed successfully.")
print("Results saved in the results folder.")
