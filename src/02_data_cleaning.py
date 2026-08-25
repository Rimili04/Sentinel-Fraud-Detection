import pandas as pd

# Load dataset
data = pd.read_csv("data/sentinel_data.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

print("\nColumns:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())


# Remove previous_behavior to avoid data leakage
data = data.drop("previous_behavior", axis=1)


# Remove user ID because it is not useful for ML
data = data.drop(data.columns[0], axis=1)


# Convert timestamp to datetime
data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    format="%d-%m-%Y %H:%M"
)


# Extract useful time features
data["hour"] = data["timestamp"].dt.hour

data["day_of_week"] = data["timestamp"].dt.dayofweek

data["is_weekend"] = data["day_of_week"].apply(
    lambda x: 1 if x >= 5 else 0
)


# Remove original timestamp
data = data.drop("timestamp", axis=1)


# Display final columns
print("\nColumns after cleaning:")
print(data.columns)


# Display first 5 rows
print("\nFirst 5 rows:")
print(data.head())


# Save cleaned dataset
data.to_csv("data/cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully!")