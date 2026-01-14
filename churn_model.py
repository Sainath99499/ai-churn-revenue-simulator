import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("../data/churn.csv")

# Encode categorical columns
encoder = LabelEncoder()
for col in ["Contract", "PaymentMethod", "TechSupport", "Churn"]:
    df[col] = encoder.fit_transform(df[col])

# Features & Target
X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Predictions
churn_prob = model.predict_proba(X)[:, 1]

# Business Logic (🔥 CRAZY PART)
df["Churn_Probability"] = churn_prob
df["Expected_Revenue_Loss"] = df["Churn_Probability"] * df["MonthlyCharges"] * 6

# Save output for Power BI
final_df = df[
    ["customerID", "tenure", "MonthlyCharges", "Churn_Probability", "Expected_Revenue_Loss"]
]

final_df.to_csv("../output/churn_predictions.csv", index=False)

print("✅ Model completed. File saved for Power BI.")
