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

## Objectives

- Analyze credit card transaction data.
- Preprocess and scale transaction features.
- Handle class imbalance using SMOTE.
- Train an LSTM Autoencoder using normal transactions.
- Calculate reconstruction error for anomaly detection.
- Identify potentially suspicious transactions.
- Evaluate the model using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Assign LOW, MEDIUM, and HIGH transaction risk levels.
- Provide single-transaction fraud risk prediction.
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
       +------+------+
       |      |      |
       v      v      v
      LOW   MEDIUM  HIGH
      RISK   RISK   RISK
      ## Risk Assessment

The system converts the anomaly score into three project-defined risk levels.

| Risk Level | Description |
|---|---|
| LOW | Low anomaly score |
| MEDIUM | Moderate anomaly score |
| HIGH | High anomaly score |

### Risk Thresholds

The thresholds are calculated from the reconstruction-error distribution of normal transactions.

- LOW threshold: approximately `1.2243`
- HIGH threshold: approximately `5.3829`

These thresholds are **project-specific experimental thresholds** and are not official banking, financial, or regulatory standards.


## Model Performance

The model was evaluated on the test dataset using multiple performance metrics.

| Metric | Result |
|---|---:|
| Accuracy | **99.43%** |
| Precision | **17.39%** |
| Recall | **61.22%** |
| F1-Score | **27.09%** |
| ROC-AUC | **96.17%** |

### Performance Interpretation

- **Accuracy: 99.43%** — overall accuracy on the test dataset.
- **Precision: 17.39%** — proportion of transactions predicted as fraudulent that were actually fraudulent.
- **Recall: 61.22%** — proportion of fraudulent transactions detected by the model.
- **F1-Score: 27.09%** — balance between precision and recall for the fraud class.
- **ROC-AUC: 96.17%** — ability of the anomaly scores to distinguish between normal and fraudulent transactions across classification thresholds.

Because the dataset is highly imbalanced, accuracy should be interpreted together with Precision, Recall, F1-Score, and ROC-AUC.


## Sample Predictions

### Normal Transaction

```text
Anomaly Score : 0.1833
Prediction    : NORMAL
Risk Level    : LOW
## Source Code Modules

### `explore_data.py`

Performs:

- Dataset exploration
- Missing-value analysis
- Duplicate checking
- Class distribution analysis
- Transaction amount analysis
- Visualization generation

### `preprocess.py`

Performs:

- Feature and target separation
- Train/test splitting
- Feature scaling
- SMOTE-based class balancing
- Saving processed datasets

### `train_autoencoder.py`

Performs:

- Normal transaction selection
- LSTM Autoencoder construction
- Model training
- Early stopping
- Model checkpointing
- Saving the trained model

### `evaluate.py`

Performs:

- Reconstruction error calculation
- Anomaly threshold calculation
- Fraud prediction
- Precision calculation
- Recall calculation
- F1-Score calculation
- ROC-AUC calculation
- Confusion matrix generation
- ROC curve generation

### `risk_assessment.py`

Performs:

- Reconstruction-error-based risk classification
- LOW/MEDIUM/HIGH risk assignment
- Risk distribution analysis

### `predict.py`

Performs:

- Single transaction input
- Feature scaling
- LSTM Autoencoder prediction
- Anomaly score calculation
- Fraud/suspicious classification
- Risk-level prediction


## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- SMOTE
- Matplotlib
- Seaborn
- Joblib


## Hardware Requirements

### Minimum Requirements

- Intel Core i5 processor or equivalent
- 8 GB RAM
- 20 GB available storage
- Windows 10/11 or Linux
- CPU-based execution supported

### Recommended

- Intel Core i5/i7 processor or equivalent
- 16 GB RAM
- SSD storage
- GPU optional

The project can be executed on a CPU-based laptop, although training time depends on the available hardware.


## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/prajwalgounalli-spec/Credit-Card-Fraud-Risk-Assessment.git