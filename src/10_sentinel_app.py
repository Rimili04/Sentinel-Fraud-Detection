import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Sentinel",
    page_icon="🛡️",
    layout="wide"
)

DATA_PATH = "data/sentinel_data.csv"
ENCODED_DATA_PATH = "data/encoded_data.csv"
MODEL_PATH = "models/random_forest_model.pkl"
THRESHOLD_PATH = "results/final_threshold.csv"
IMPORTANCE_PATH = "results/feature_importance.csv"

data = pd.read_csv(DATA_PATH)
encoded_data = pd.read_csv(ENCODED_DATA_PATH)
model = joblib.load(MODEL_PATH)
threshold_data = pd.read_csv(THRESHOLD_PATH)
feature_importance = pd.read_csv(IMPORTANCE_PATH)

threshold = float(threshold_data.iloc[0]["best_threshold"])

model_features = [
    "activity",
    "amount",
    "device",
    "location",
    "frequency",
    "failed_attempts",
    "hour",
    "day_of_week",
    "is_weekend"
]

categorical_columns = [
    "activity",
    "device",
    "location"
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    encoder.fit(data[column].astype(str))
    encoders[column] = encoder

st.title("🛡️ Sentinel")
st.subheader("Intelligent Transaction Risk Detection")

st.write(
    "Enter transaction details below to analyse the transaction "
    "and estimate its fraud risk."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    activity = st.selectbox(
        "Activity",
        sorted(data["activity"].astype(str).unique())
    )

    amount = st.number_input(
        "Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    device = st.selectbox(
        "Device",
        sorted(data["device"].astype(str).unique())
    )

with col2:
    location = st.selectbox(
        "Location",
        sorted(data["location"].astype(str).unique())
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0,
        value=1,
        step=1
    )

    failed_attempts = st.number_input(
        "Failed Attempts",
        min_value=0,
        value=0,
        step=1
    )

with col3:
    hour = st.slider(
        "Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    day_of_week = st.slider(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2
    )

    is_weekend = 1 if day_of_week >= 5 else 0

st.divider()

analyse = st.button(
    "🔍 Analyse Transaction",
    use_container_width=True
)

if analyse:
    try:
        input_data = pd.DataFrame({
            "activity": [activity],
            "amount": [amount],
            "device": [device],
            "location": [location],
            "frequency": [frequency],
            "failed_attempts": [failed_attempts],
            "hour": [hour],
            "day_of_week": [day_of_week],
            "is_weekend": [is_weekend]
        })

        # Encode categorical values
        for column in categorical_columns:
            input_data[column] = encoders[column].transform(
                input_data[column].astype(str)
            )

        input_data = input_data[model_features]

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        fraud_prediction = probability >= threshold

        if probability >= threshold:
            risk_level = "HIGH RISK"
        elif probability >= threshold * 0.75:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"

        # Save the exact analysed transaction
        st.session_state["analyzed_transaction"] = {
            "activity": activity,
            "amount": amount,
            "device": device,
            "location": location,
            "frequency": frequency,
            "failed_attempts": failed_attempts,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "probability": probability,
            "fraud_prediction": fraud_prediction,
            "risk_level": risk_level,
            "threshold": threshold
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")


if "analyzed_transaction" in st.session_state:

    transaction = st.session_state["analyzed_transaction"]

    probability = transaction["probability"]
    threshold = transaction["threshold"]
    risk_level = transaction["risk_level"]
    fraud_prediction = transaction["fraud_prediction"]

    st.divider()

    st.subheader("Sentinel Analysis")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with metric2:
        st.metric(
            "Detection Threshold",
            f"{threshold * 100:.2f}%"
        )

    with metric3:
        st.metric(
            "Risk Level",
            risk_level
        )

    st.divider()

    if fraud_prediction:

        st.error("🚨 HIGH RISK TRANSACTION")

        st.write(
            "This transaction has been classified as potentially fraudulent."
        )

    elif probability >= threshold * 0.75:

        st.warning("⚠️ MEDIUM RISK TRANSACTION")

        st.write(
            "This transaction shows suspicious characteristics "
            "and may require additional verification."
        )

    else:

        st.success("✅ LOW RISK TRANSACTION")

        st.write(
            "This transaction appears to have a relatively low fraud risk."
        )

    st.divider()

    st.subheader("Transaction Details")

    detail1, detail2, detail3 = st.columns(3)

    with detail1:
        st.write(
            f"**Activity:** {transaction['activity']}"
        )

        st.write(
            f"**Amount:** ₹{transaction['amount']:,.2f}"
        )

        st.write(
            f"**Device:** {transaction['device']}"
        )

    with detail2:
        st.write(
            f"**Location:** {transaction['location']}"
        )

        st.write(
            f"**Frequency:** {transaction['frequency']}"
        )

        st.write(
            f"**Failed Attempts:** {transaction['failed_attempts']}"
        )

    with detail3:
        st.write(
            f"**Hour:** {transaction['hour']:02d}:00"
        )

        st.write(
            f"**Day of Week:** {transaction['day_of_week']}"
        )

        st.write(
            f"**Weekend:** {'Yes' if transaction['is_weekend'] else 'No'}"
        )

    st.divider()

    st.subheader("Sentinel Explanation")

    importance_data = feature_importance.copy()

    if "feature" in importance_data.columns and "importance" in importance_data.columns:

        importance_data = importance_data.sort_values(
            "importance",
            ascending=False
        )

        top_features = importance_data.head(5)

        st.write("Major factors considered by the model:")

        for _, row in top_features.iterrows():

            feature = row["feature"]
            importance = float(row["importance"])

            if feature == "amount":
                explanation = (
                    f"Transaction amount has a strong influence "
                    f"on the prediction ({importance:.2f})."
                )

            elif feature == "location":
                explanation = (
                    f"Transaction location contributes significantly "
                    f"to the prediction ({importance:.2f})."
                )

            elif feature == "hour":
                explanation = (
                    f"Transaction time is an important factor "
                    f"({importance:.2f})."
                )

            elif feature == "frequency":
                explanation = (
                    f"Transaction frequency influences the risk score "
                    f"({importance:.2f})."
                )

            elif feature == "failed_attempts":
                explanation = (
                    f"Failed attempts are considered an important "
                    f"risk indicator ({importance:.2f})."
                )

            elif feature == "activity":
                explanation = (
                    f"Transaction activity contributes to the prediction "
                    f"({importance:.2f})."
                )

            elif feature == "device":
                explanation = (
                    f"Device type contributes to the prediction "
                    f"({importance:.2f})."
                )

            elif feature == "day_of_week":
                explanation = (
                    f"Day of week contributes to the prediction "
                    f"({importance:.2f})."
                )

            elif feature == "is_weekend":
                explanation = (
                    f"Weekend status contributes to the prediction "
                    f"({importance:.2f})."
                )

            else:
                explanation = (
                    f"{feature} contributes to the model prediction "
                    f"({importance:.2f})."
                )

            st.write(f"• {explanation}")

    else:

        st.write(
            "Feature importance information is available, "
            "but the expected columns were not found."
        )

    st.divider()

    st.subheader("Recommendation")

    if fraud_prediction:

        st.error(
            "🚨 Additional verification is recommended before "
            "allowing this transaction."
        )

    elif probability >= threshold * 0.75:

        st.warning(
            "⚠️ Consider additional verification before proceeding."
        )

    else:

        st.success(
            "✅ The transaction can proceed under normal monitoring."
        )

    st.divider()

    st.caption(
        f"Sentinel model threshold: {threshold:.2f}"
    )