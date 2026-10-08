import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CardioGuard", page_icon="🫀", layout="wide")

st.title("🫀 CardioGuard: Heart Attack Risk Diagnostic")
st.write("Enter patient clinical parameters below for real-time risk diagnostic.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 100, 55)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 130)

with col2:
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)

with col3:
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored by Flourosopy (0-3)", [0, 1, 2, 3])

# Patient dataframe matching exact feature list
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca']
patient_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]], columns=feature_names)

if st.button("Run Diagnostic Screening"):
    try:
        model = joblib.load("cardio.plk")
        explainer = joblib.load("sharp_explainer.plk")

        # Force alignment with model booster features
        booster_features = model.get_booster().feature_names
        if booster_features:
            for col in booster_features:
                if col not in patient_data.columns:
                    patient_data[col] = 0
            patient_data = patient_data[booster_features]

        risk_prob = model.predict_proba(patient_data)[0][1] * 100

        st.subheader("Diagnostic Results")
        if risk_prob > 50:
            st.error(f"⚠️ High Heart Attack Risk Detected: **{risk_prob:.1f}%**")
        else:
            st.success(f"✅ Low Heart Attack Risk Detected: **{risk_prob:.1f}%**")

        shap_values = explainer(patient_data)
        patient_impact = pd.Series(shap_values.values[0], index=patient_data.columns).sort_values(ascending=False)

        st.subheader("Primary Biological Drivers")
        for feature, impact in patient_impact.head(3).items():
            if impact > 0:
                st.write(f"- **{feature.upper()}** (Value: {patient_data[feature].values[0]})")

    except Exception as e:
        st.error(f"Error loading prediction model: {e}")