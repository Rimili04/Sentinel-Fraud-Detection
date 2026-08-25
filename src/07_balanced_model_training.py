import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


data = pd.read_csv("data/encoded_data.csv")

print("Encoded dataset loaded successfully!")
print("Dataset shape:", data.shape)


X = data.drop("risk_label", axis=1)
y = data["risk_label"]


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


print("\nTraining balanced Random Forest model...")


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1
)


model.fit(X_train, y_train)


print("Balanced model training completed!")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)


print("\nBalanced Model Results:")

print("Accuracy:", round(accuracy, 4))
print("Fraud Precision:", round(precision, 4))
print("Fraud Recall:", round(recall, 4))
print("Fraud F1-score:", round(f1, 4))


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


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
    "results/balanced_feature_importance.csv",
    index=False
)


joblib.dump(
    model,
    "models/balanced_random_forest_model.pkl"
)


print("\nBalanced model saved successfully!")
print("Feature importance saved successfully!")