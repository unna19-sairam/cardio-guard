import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="CardioGuard", page_icon="🫀", layout="wide")

st.title("🫀 CardioGuard: Heart Attack Risk & Biological Root-Cause Analysis")
st.write("Enter patient clinical parameters below for real-time risk diagnostic.")

# Input fields sidebar/columns
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

# Preprocessing patient input into DataFrame matching training data
patient_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]],
                            columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca'])

medical_causes = {
    'oldpeak': "Myocardial Ischemia (ST Segment Stress)",
    'ca': "Coronary Artery Occlusion (Blocked Major Vessels)",
    'chol': "Atherosclerotic Risk (Elevated Serum Cholesterol)",
    'trestbps': "Hypertension Strain (High Blood Pressure)",
    'thalach': "Reduced Cardiac Reserve Capacity (Low Max Heart Rate)",
    'fbs': "Metabolic/Diabetic Stress (High Fasting Blood Sugar)",
    'exang': "Exercise-Induced Angina (Chest Pain during Exercise)"
}

if st.button("Run Diagnostic Screening"):
    try:
        model = joblib.load("cardio_model.pkl")
        explainer = joblib.load("shap_explainer.pkl")
        
        # Risk probability
        risk_prob = model.predict_proba(patient_data)[0][1] * 100
        
        st.subheader("Diagnostic Results")
        if risk_prob > 50:
            st.error(f"⚠️ High Heart Attack Risk Detected: **{risk_prob:.1f}%**")
        else:
            st.success(f"✅ Low Heart Attack Risk Detected: **{risk_prob:.1f}%**")
            
        # SHAP Etiology Analysis
        shap_values = explainer(patient_data)
        patient_impact = pd.Series(shap_values.values[0], index=patient_data.columns).sort_values(ascending=False)
        
        st.subheader("Primary Biological Root Causes")
        top_causes = []
        for feature, impact in patient_impact.items():
            if impact > 0 and feature in medical_causes:
                top_causes.append(f"- **{medical_causes[feature]}** (Measured Value: {patient_data[feature].values[0]})")
                if len(top_causes) == 3:
                    break
        
        if top_causes:
            for cause in top_causes:
                st.write(cause)
        else:
            st.write("No major elevated biological risk drivers identified.")
            
    except FileNotFoundError:
        st.error("Missing model files! Run 'python train.py' in your folder first to generate 'cardio_model.pkl' and 'shap_explainer.pkl'.")