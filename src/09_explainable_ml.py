import pandas as pd
import joblib
import matplotlib.pyplot as plt


data = pd.read_csv("data/encoded_data.csv")

model = joblib.load("models/random_forest_model.pkl")

X = data.drop("risk_label", axis=1)


feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("Explainable ML loaded successfully!")

print("\nFeature Importance:")
print(feature_importance)


feature_importance.to_csv(
    "results/explainable_feature_importance.csv",
    index=False
)


plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Sentinel - Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png"
)

plt.show()


print("\nExplainable ML results saved successfully!")
print("Feature importance chart saved successfully!")