import pandas as pd

# Load dataset
data = pd.read_csv("data/sentinel_data.csv")

# Basic information
print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))

print("\nColumn names:")
print(data.columns)

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())

print("\nRisk label distribution:")
print(data["risk_label"].value_counts())