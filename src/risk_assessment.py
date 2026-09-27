import numpy as np

ERRORS_PATH = "results/reconstruction_errors.npy"
LABELS_PATH = "results/test_labels.npy"

errors = np.load(ERRORS_PATH)
labels = np.load(LABELS_PATH)

normal_errors = errors[labels == 0]

low_threshold = np.percentile(normal_errors, 95)
high_threshold = np.percentile(normal_errors, 99.5)

print("=" * 50)
print("TRANSACTION RISK ASSESSMENT")
print("=" * 50)

print(f"Low Risk Threshold    : {low_threshold:.4f}")
print(f"High Risk Threshold   : {high_threshold:.4f}")

def assess_risk(anomaly_score):
    if anomaly_score <= low_threshold:
        return "LOW"
    elif anomaly_score <= high_threshold:
        return "MEDIUM"
    else:
        return "HIGH"

risk_levels = np.array([
    assess_risk(score)
    for score in errors
])

print()
print("Risk Distribution:")
print("LOW    :", np.sum(risk_levels == "LOW"))
print("MEDIUM :", np.sum(risk_levels == "MEDIUM"))
print("HIGH   :", np.sum(risk_levels == "HIGH"))

np.save(
    "results/risk_levels.npy",
    risk_levels
)

with open("results/risk_thresholds.txt", "w") as file:
    file.write(f"LOW_THRESHOLD={low_threshold}\n")
    file.write(f"HIGH_THRESHOLD={high_threshold}\n")

print()
print("Risk assessment completed successfully.")
print("Risk levels saved in results/risk_levels.npy")
