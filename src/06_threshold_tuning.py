import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
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


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = joblib.load("models/random_forest_model.pkl")

print("\nModel loaded successfully!")


probabilities = model.predict_proba(X_test)[:, 0]


thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


results = []


print("\nThreshold Results:")
print("-" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1-score':<12}"
)


for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    predictions = 1 - predictions

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0
    )

    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    )

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.3f}"
        f"{precision:<12.3f}"
        f"{recall:<12.3f}"
        f"{f1:<12.3f}"
    )


results_df = pd.DataFrame(results)


best_result = results_df.loc[
    results_df["f1_score"].idxmax()
]


best_threshold = best_result["threshold"]


print("\nBest Threshold:")
print(best_threshold)

print("\nBest Fraud Detection Performance:")

print(
    "Accuracy:",
    round(best_result["accuracy"], 3)
)

print(
    "Precision:",
    round(best_result["precision"], 3)
)

print(
    "Recall:",
    round(best_result["recall"], 3)
)

print(
    "F1-score:",
    round(best_result["f1_score"], 3)
)


best_predictions = (
    probabilities >= best_threshold
).astype(int)

best_predictions = 1 - best_predictions


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)


results_df.to_csv(
    "results/threshold_results.csv",
    index=False
)


print("\nThreshold results saved successfully!")