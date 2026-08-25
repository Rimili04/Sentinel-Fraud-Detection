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


model = joblib.load(
    "models/random_forest_model.pkl"
)

print("Original Random Forest loaded successfully!")


probabilities = model.predict_proba(X_test)

fraud_class_index = list(model.classes_).index(0)

fraud_probabilities = probabilities[:, fraud_class_index]


thresholds = [
    round(x / 100, 2)
    for x in range(5, 51)
]


results = []


print("\nFinal Threshold Results:")
print("-" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1-score':<12}"
)


for threshold in thresholds:

    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    predictions = 1 - predictions

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

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


print("\nBest Final Threshold:")
print(best_threshold)


print("\nBest Final Performance:")

print(
    "Accuracy:",
    round(best_result["accuracy"], 4)
)

print(
    "Fraud Precision:",
    round(best_result["precision"], 4)
)

print(
    "Fraud Recall:",
    round(best_result["recall"], 4)
)

print(
    "Fraud F1-score:",
    round(best_result["f1_score"], 4)
)


best_predictions = (
    fraud_probabilities >= best_threshold
).astype(int)

best_predictions = 1 - best_predictions


print("\nFinal Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)


results_df.to_csv(
    "results/final_threshold_results.csv",
    index=False
)


final_threshold = pd.DataFrame({
    "best_threshold": [best_threshold],
    "accuracy": [best_result["accuracy"]],
    "fraud_precision": [best_result["precision"]],
    "fraud_recall": [best_result["recall"]],
    "fraud_f1_score": [best_result["f1_score"]]
})


final_threshold.to_csv(
    "results/final_threshold.csv",
    index=False
)


print("\nFinal threshold results saved successfully!")