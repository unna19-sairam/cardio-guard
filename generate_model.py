import joblib
import pandas as pd
import shap
from xgboost import XGBClassifier

# Reliable fallback URL for the heart disease dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
df = pd.read_csv(url)

# 12 features matching app.py
feature_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
]
X = df[feature_names]
y = df["target"]

model = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
)
model.fit(X, y)

explainer = shap.TreeExplainer(model)

# Save models with standard and legacy filenames for maximum compatibility
joblib.dump(model, "cardio_model.pkl")
joblib.dump(model, "cardio.plk")

joblib.dump(explainer, "shap_explainer.pkl")
joblib.dump(explainer, "sharp_explainer.plk")

print("New risk prediction model (cardio_model.pkl) and cause explainer model (shap_explainer.pkl) generated successfully!")