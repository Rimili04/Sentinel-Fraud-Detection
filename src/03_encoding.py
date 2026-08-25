import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/cleaned_data.csv")

print("Cleaned dataset loaded successfully!")
print("Dataset shape:", data.shape)

categorical_columns = [
    "activity",
    "device",
    "location",
    "risk_label"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])

    print(f"\nEncoded column: {column}")
    print(data[column].value_counts())

print("\nAfter encoding:")
print(data.head())

print("\nData types:")
print(data.dtypes)

data.to_csv(
    "data/encoded_data.csv",
    index=False
)

print("\nEncoded dataset saved successfully!")