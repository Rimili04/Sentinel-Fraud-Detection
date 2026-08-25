import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

data = pd.read_csv("data/encoded_data.csv")

print("Encoded dataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nColumns:")
print(data.columns)

X = data.drop("risk_label", axis=1)
y = data["risk_label"]

print("\nFeatures:")
print(X.columns)

print("\nTarget distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

feature_importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("Trained model saved successfully!")