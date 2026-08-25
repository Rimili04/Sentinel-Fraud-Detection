 🛡️ Sentinel — Intelligent Transaction Risk Detection

> An explainable machine-learning system for detecting potentially fraudulent transactions, estimating fraud probability, classifying risk, and providing interpretable security recommendations.

---

## 🚀 Overview

**Sentinel** is an end-to-end machine learning project designed to assess the risk of individual financial transactions.

The system takes transaction characteristics such as activity, amount, location, device, timing, transaction frequency, and failed attempts, processes them through a trained machine-learning pipeline, and produces a **fraud probability and risk assessment**.

Instead of returning only a simple `Fraud / Not Fraud` prediction, Sentinel provides:

- 🎯 Fraud probability
- 🚨 Risk classification
- 🎚️ Optimized detection threshold
- 🧠 Explainable ML insights
- 📊 Feature importance
- 💡 Risk-based recommendations
- 🖥️ Interactive web interface

The complete workflow is:

```text
Transaction Details
       ↓
Data Preprocessing
       ↓
Feature Encoding
       ↓
Machine Learning Model
       ↓
Fraud Probability
       ↓
Detection Threshold
       ↓
Risk Classification
       ↓
Explainable ML
       ↓
Risk Recommendation
🎯 Problem Statement

Financial institutions process a large number of transactions, making it difficult to manually identify potentially fraudulent activity.

Traditional rule-based approaches may struggle to capture complex relationships between transaction characteristics.

Sentinel uses machine learning to identify patterns associated with fraudulent transactions and convert the model's prediction into an understandable risk assessment.

The system is designed around three key questions:

Is the transaction suspicious?

How risky is it?

What factors contributed to the prediction?

✨ Key Features
🔍 Transaction Risk Detection

Sentinel analyzes multiple transaction attributes, including:

Activity type
Transaction amount
Location
Device
Transaction hour
Transaction frequency
Failed attempts
Day of week
Weekend status
🎯 Fraud Probability

The trained model generates a probability score representing the estimated likelihood of fraudulent activity.

Example:
Fraud Probability
94.50%
This provides more information than a simple binary prediction.

🚨 Risk Classification
The predicted probability is compared against the selected detection threshold and converted into a risk category.

Risk Level	Interpretation
🔴 HIGH RISK	Strong indication of suspicious activity
🟠 MEDIUM RISK	Elevated risk requiring additional attention
🟢 LOW RISK	Relatively low predicted risk
🎚️ Threshold-Based Detection

Fraud detection does not always perform best using the default 0.50 classification threshold.
Sentinel includes a dedicated threshold-tuning stage to determine a suitable operating threshold for the trained model.
Example:
Fraud Probability     94.50%
Detection Threshold   39.00%

Result: HIGH RISK
Because:
94.50% > 39.00%
the transaction is classified as high risk.

🧠 Explainable Machine Learning
A machine-learning prediction is much more useful when the user can understand the factors behind it.
Sentinel includes feature-importance and explainable ML analysis to identify the transaction characteristics that contribute to the model's decisions.
This helps reduce the "black box" nature of machine-learning predictions.

💡 Risk-Based Recommendation
Sentinel converts the model's output into an understandable recommendation.
For example, a high-risk transaction may produce:
HIGH RISK

Additional verification is recommended
before allowing this transaction.

This makes the system a risk decision-support tool, rather than only a prediction model.

🤖 Machine Learning
Random Forest Classifier
The primary model used by Sentinel is a Random Forest Classifier.
Random Forest is well suited for structured/tabular transaction data and can model nonlinear relationships between transaction characteristics and fraud risk.
The project also includes a balanced-model training stage to improve handling of class imbalance.

🔬 Machine Learning Pipeline
The project follows a complete machine-learning workflow:
Raw Dataset
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Feature Encoding
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Threshold Tuning
     ↓
Balanced Model Training
     ↓
Final Threshold Selection
     ↓
Explainable ML
     ↓
Streamlit Application

Each stage is implemented separately to make the workflow easier to understand, evaluate, and reproduce.

📊 Model Development
1. Data Understanding
The dataset is explored to understand its structure, features, distributions, and target variable.

2. Data Cleaning
The data is prepared for machine learning by handling data-quality issues and preparing the required features.

3. Feature Encoding
Categorical transaction attributes are transformed into numerical representations suitable for machine-learning models.

4. Model Training
A Random Forest classification model is trained using the prepared transaction data.

5. Model Evaluation
The trained model is evaluated using appropriate classification metrics.

6. Threshold Tuning
Different classification thresholds are evaluated to determine a suitable fraud-detection threshold.

7. Balanced Model Training
A balanced training approach is used to address the challenges associated with imbalanced fraud datasets.

8. Final Threshold Selection
The final threshold used by the application is selected based on the threshold analysis.

9. Explainable ML
Feature importance is analyzed to understand which transaction characteristics contribute most strongly to the model.

10. Application Integration
The trained model, selected threshold, and explainability results are integrated into the Sentinel Streamlit application.

🖥️ Interactive Web Application
Sentinel provides an interactive web interface built with Streamlit.
Users can enter transaction details such as:

Activity
Location
Amount
Hour
Frequency
Day of week
Device
Failed attempts

After selecting Analyse Transaction, Sentinel processes the transaction and displays the resulting risk assessment.

📌 Example

A transaction may produce:

┌─────────────────────────────────────┐
│        SENTINEL ANALYSIS            │
├─────────────────────────────────────┤
│                                     │
│ Fraud Probability     94.50%        │
│ Detection Threshold   39.00%        │
│ Risk Level            HIGH RISK     │
│                                     │
└─────────────────────────────────────┘

The application then provides the corresponding recommendation and explainable information.

🏗️ System Architecture
                    ┌─────────────────────┐
                    │  Transaction Input  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Encoding    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Random Forest      │
                    │      Model          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Fraud Probability   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Detection Threshold │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
          HIGH RISK        MEDIUM RISK       LOW RISK
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Explainable ML      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Recommendation      │
                    └─────────────────────┘
📁 Project Structure
Sentinel/
│
├── data/
│   ├── sentinel_data.csv
│   ├── cleaned_data.csv
│   └── encoded_data.csv
│
├── models/
│   ├── random_forest_model.pkl
│   └── balanced_random_forest_model.pkl
│
├── notebooks/
│
├── results/
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── explainable_feature_importance.csv
│   ├── model_evaluation.csv
│   ├── final_threshold.csv
│   ├── final_threshold_results.csv
│   └── threshold_results.csv
│
├── src/
│   ├── 01_data_understanding.py
│   ├── 02_data_cleaning.py
│   ├── 03_encoding.py
│   ├── 04_model_training.py
│   ├── 05_model_evaluation.py
│   ├── 06_threshold_tuning.py
│   ├── 07_balanced_model_training.py
│   ├── 08_final_threshold_selection.py
│   ├── 09_explainable_ml.py
│   └── 10_sentinel_app.py
│
├── .gitignore
└── README.md
🛠️ Tech Stack
Technology	Purpose
Python	Core development
Pandas	Data manipulation and preprocessing
NumPy	Numerical computation
Scikit-learn	Machine learning
Joblib	Model persistence
Streamlit	Interactive web application
Matplotlib	Visualization
Git	Version control
GitHub	Source-code management
🚀 Installation
1. Clone the repository
git clone https://github.com/Rimili04/Sentinel-Fraud-Detection.git
cd Sentinel-Fraud-Detection
2. Create a virtual environment
Windows
python -m venv venv
venv\Scripts\activate
macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install pandas numpy scikit-learn streamlit matplotlib joblib
4. Run the application
streamlit run src/10_sentinel_app.py

The application will be available locally at:

http://localhost:8501
📈 Project Outputs

The project generates several outputs during the machine-learning workflow, including:

Model evaluation results
Feature importance
Explainable ML results
Threshold tuning results
Final detection threshold
Trained model artifacts

These outputs are organized inside the:

models/
results/

directories.

🔐 Data Privacy

The project is intended for development, experimentation, and demonstration.

Never upload real or confidential financial information.

Do not use:

Real customer transaction records
Bank account information
Card numbers
Personally identifiable information
Confidential banking data
Proprietary financial datasets

Use synthetic, anonymized, or publicly available datasets for experimentation.

⚠️ Disclaimer

Sentinel is a machine-learning prototype and decision-support system.

A high-risk prediction does not by itself prove that a transaction is fraudulent.

In a real-world financial environment, machine-learning predictions would typically be combined with additional security controls, business rules, verification mechanisms, monitoring systems, and human investigation.

🎓 What This Project Demonstrates

Sentinel demonstrates practical implementation of:

Machine Learning
Binary Classification
Fraud Risk Detection
Data Preprocessing
Feature Engineering
Categorical Encoding
Random Forest Classification
Class Imbalance Handling
Model Evaluation
Threshold Optimization
Explainable Machine Learning
Feature Importance Analysis
Risk Classification
Probability-Based Prediction
Model Serialization
Streamlit Application Development
End-to-End ML Pipeline
Git & GitHub
🌟 Why Sentinel?

A basic fraud-detection project may stop at:
Input → Fraud / Not Fraud
Sentinel goes further:
Transaction
     ↓
ML Prediction
     ↓
Fraud Probability
     ↓
Threshold Comparison
     ↓
Risk Level
     ↓
Explainable Factors
     ↓
Security Recommendation

This approach makes the model output easier to interpret and more useful for transaction-risk analysis.

👩‍💻 Author
Rimili Dutta
Repository: Sentinel-Fraud-Detection
Site link - http://localhost:8501/
