# Predictive Deep Learning Model for Credit Card Fraud Detection and Risk Assessment

## Project Overview

Credit card fraud is a major challenge in digital financial transactions because fraudulent transactions are rare compared with normal transactions.

This project develops a deep learning-based fraud detection system using an **LSTM Autoencoder** for anomaly detection. The model learns the reconstruction patterns of normal credit card transactions and uses reconstruction error to identify potentially suspicious transactions.

The system also performs data preprocessing, feature scaling, class imbalance handling using **SMOTE**, model evaluation, anomaly detection, and transaction risk assessment.

The system categorizes transactions into:

- LOW Risk
- MEDIUM Risk
- HIGH Risk

This project is developed for academic and educational purposes.

---

## Dataset

This project uses the **Credit Card Fraud Detection dataset** containing anonymized credit card transactions.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total Transactions | 284,807 |
| Normal Transactions | 284,315 |
| Fraudulent Transactions | 492 |
| Input Features | 30 |
| Target Variable | `Class` |
| Normal Class | 0 |
| Fraud Class | 1 |

### Features

The dataset contains:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

The `V1` to `V28` features are anonymized transaction features.

The original dataset is **not included in this GitHub repository** because of its size and distribution terms.

---

## Objectives

The main objectives of this project are:

1. To analyze credit card transaction data.
2. To preprocess and normalize transaction features.
3. To handle class imbalance using SMOTE.
4. To train an LSTM Autoencoder using normal transaction data.
5. To calculate reconstruction error for anomaly detection.
6. To identify potentially suspicious transactions.
7. To evaluate the model using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
8. To develop a transaction risk assessment system.
9. To classify transactions into LOW, MEDIUM, and HIGH risk categories.
10. To provide single-transaction fraud risk prediction.

---

## System Architecture

```text
Credit Card Transaction Dataset
              |
              v
       Data Exploration
              |
              v
      Data Preprocessing
              |
              v
       Feature Scaling
              |
              v
       Train/Test Split
              |
              v
      SMOTE Class Balancing
              |
              v
   Normal Transaction Selection
              |
              v
       LSTM Autoencoder
              |
              v
      Reconstruction Error
              |
              v
       Anomaly Detection
              |
              v
        Risk Assessment
              |
        +-----+-----+
        |     |     |
        v     v     v
       LOW  MEDIUM HIGH
       RISK   RISK  RISK
Risk Assessment

The system converts the anomaly score into three project-defined risk levels.

Risk Level	Description
LOW	Low anomaly score
MEDIUM	Moderate anomaly score
HIGH	High anomaly score
Risk Thresholds

The thresholds are calculated from the reconstruction-error distribution of normal transactions.

LOW threshold: approximately 1.2243
HIGH threshold: approximately 5.3829

These thresholds are project-specific experimental thresholds and are not official banking, financial, or regulatory standards.

Model Performance

The model was evaluated on the test dataset using multiple performance metrics.

Metric	Result
Accuracy	99.43%
Precision	17.39%
Recall	61.22%
F1-Score	27.09%
ROC-AUC	96.17%
Performance Interpretation
Accuracy: 99.43% — overall accuracy on the test dataset.
Precision: 17.39% — proportion of transactions predicted as fraudulent that were actually fraudulent.
Recall: 61.22% — proportion of fraudulent transactions detected by the model.
F1-Score: 27.09% — balance between precision and recall for the fraud class.
ROC-AUC: 96.17% — ability of the anomaly scores to distinguish between normal and fraudulent transactions across classification thresholds.

Because the dataset is highly imbalanced, accuracy should be interpreted together with Precision, Recall, F1-Score, and ROC-AUC.

Sample Predictions
Normal Transaction
Anomaly Score : 0.1833
Prediction    : NORMAL
Risk Level    : LOW
Fraud-Labeled Transaction
Anomaly Score : 2.2269
Prediction    : SUSPICIOUS
Risk Level    : MEDIUM

The second example corresponds to a transaction originally labeled as fraud in the dataset. The anomaly-detection model assigns its risk level based on reconstruction error.

Source Code Modules
explore_data.py

Performs:

Dataset exploration
Missing-value analysis
Duplicate checking
Class distribution analysis
Transaction amount analysis
Visualization generation
preprocess.py

Performs:

Feature and target separation
Train/test splitting
Feature scaling
SMOTE-based class balancing
Saving processed datasets
train_autoencoder.py

Performs:

Normal transaction selection
LSTM Autoencoder construction
Model training
Early stopping
Model checkpointing
Saving the trained model
evaluate.py

Performs:

Reconstruction error calculation
Anomaly threshold calculation
Fraud prediction
Precision calculation
Recall calculation
F1-Score calculation
ROC-AUC calculation
Confusion matrix generation
ROC curve generation
risk_assessment.py

Performs:

Reconstruction-error-based risk classification
LOW/MEDIUM/HIGH risk assignment
Risk distribution analysis
predict.py

Performs:

Single transaction input
Feature scaling
LSTM Autoencoder prediction
Anomaly score calculation
Fraud/suspicious classification
Risk-level prediction
Project Structure
Credit-Card-Fraud-Risk-Assessment/
│
├── dataset/
│   └── creditcard.csv
│
├── models/
│   └── lstm_autoencoder.keras
│
├── results/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── threshold.txt
│   └── risk_thresholds.txt
│
├── src/
│   ├── explore_data.py
│   ├── preprocess.py
│   ├── train_autoencoder.py
│   ├── evaluate.py
│   ├── risk_assessment.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt

The dataset, trained model, processed data, and generated .npy files are excluded from GitHub according to the project's .gitignore configuration where applicable.

Technologies Used
Python
TensorFlow
Keras
NumPy
Pandas
Scikit-learn
Imbalanced-learn
SMOTE
Matplotlib
Seaborn
Joblib
Git
GitHub
Hardware Requirements
Minimum Requirements
Intel Core i5 processor or equivalent
8 GB RAM
20 GB available storage
Windows 10/11 or Linux
CPU-based execution supported
Recommended
Intel Core i5/i7 processor or equivalent
16 GB RAM
SSD storage
GPU optional

The project can be executed on a CPU-based laptop, although training time depends on the available hardware.

Installation
1. Clone the Repository
git clone https://github.com/prajwalgounalli-spec/Credit-Card-Fraud-Risk-Assessment.git
cd Credit-Card-Fraud-Risk-Assessment
2. Create a Virtual Environment
python -m venv .venv
3. Activate the Virtual Environment

For Windows PowerShell:

.venv\Scripts\Activate.ps1

For Windows Command Prompt:

.venv\Scripts\activate
4. Install Required Packages
pip install -r requirements.txt
Dataset Setup

Place the downloaded dataset at:

Credit-Card-Fraud-Risk-Assessment/
└── dataset/
    └── creditcard.csv

The expected file name is:

creditcard.csv

The dataset is not included in this repository because of its large file size and distribution considerations.

How to Run the Project

Run the project modules in the following order.

Step 1: Explore the Dataset
python src/explore_data.py

This performs basic dataset exploration and generates data distribution visualizations.

Step 2: Preprocess the Dataset
python src/preprocess.py

This performs feature preprocessing, scaling, train-test splitting, and SMOTE-based class balancing.

Step 3: Train the LSTM Autoencoder
python src/train_autoencoder.py

The autoencoder is trained using normal transactions and the trained model is saved in the models folder.

Step 4: Evaluate the Model
python src/evaluate.py

This calculates:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
ROC Curve
Step 5: Perform Risk Assessment
python src/risk_assessment.py

This categorizes transactions into:

LOW
MEDIUM
HIGH

risk levels based on reconstruction-error thresholds.

Step 6: Predict a New Transaction
python src/predict.py

The program accepts transaction features and produces:

Anomaly Score
Prediction
Risk Level
Generated Results

The project generates:

Dataset distribution visualizations
Transaction amount visualization
Confusion matrix
ROC curve
Reconstruction errors
Fraud predictions
Risk levels
Risk thresholds

Generated result files are stored in the results directory.

Project Limitations
The dataset contains highly imbalanced transaction classes.
The LSTM Autoencoder is trained primarily on normal transactions for anomaly detection.
Reconstruction-error thresholds are specific to the current dataset and model.
The model may generate false positives for unusual but legitimate transactions.
The model may not detect every type of real-world fraud pattern.
The dataset contains anonymized features, which limits direct interpretation of individual transaction attributes.
The project is developed and evaluated using a publicly available historical dataset.
The current implementation is intended for academic and demonstration purposes.
Future Enhancements
Develop a real-time transaction monitoring system.
Integrate additional transaction and customer behavioral features.
Compare LSTM Autoencoder performance with other deep learning models.
Optimize anomaly thresholds using validation data.
Add explainable AI techniques for transaction-level analysis.
Develop a web-based dashboard for fraud monitoring.
Add database integration for storing transaction risk information.
Evaluate the system on additional fraud detection datasets.
Deploy the trained model as an API for real-time prediction.
Conclusion

This project demonstrates a deep learning-based approach for credit card fraud detection and transaction risk assessment using an LSTM Autoencoder.

The system learns normal transaction patterns, calculates reconstruction error for anomaly detection, evaluates model performance, and assigns LOW, MEDIUM, and HIGH transaction risk levels.

The project combines data preprocessing, class imbalance handling, anomaly detection, deep learning, model evaluation, and risk assessment into a complete fraud detection workflow.


Prajwal Gounalli

M.Tech Data Science

GitHub: prajwalgounalli-spec