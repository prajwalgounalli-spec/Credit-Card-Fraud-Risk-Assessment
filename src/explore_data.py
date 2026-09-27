import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "dataset/creditcard.csv"
RESULTS_DIR = "results"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("CREDIT CARD FRAUD DATASET ANALYSIS")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

print("\nTransaction Amount Statistics:")
print(df["Amount"].describe())

# Class distribution plot
plt.figure(figsize=(7, 5))
sns.countplot(x="Class", data=df)
plt.title("Credit Card Transaction Class Distribution")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/class_distribution.png", dpi=300)
plt.close()

# Transaction amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Amount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/amount_distribution.png", dpi=300)
plt.close()

print("\nExploration completed successfully.")
print("Charts saved in the results folder.")
