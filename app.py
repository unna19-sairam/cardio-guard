import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="CardioGuard", page_icon="🫀", layout="wide"
)

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_screening" not in st.session_state:
    st.session_state.last_screening = None

# Login Page
if not st.session_state.authenticated:
    st.title("🫀 CardioGuard Portal Login")
    tab1, tab2 = st.tabs(["Patient Login", "Doctor Login"])
    
    with tab1:
        st.subheader("Patient Access")
        p_id = st.text_input("Patient ID", value="patient123")
        p_pass = st.text_input("Password", type="password", value="password", key="p_pass")
        if st.button("Log In as Patient"):
            if p_id and p_pass:
                st.session_state.authenticated = True
                st.session_state.user_role = "Patient"
                st.rerun()

    with tab2:
        st.subheader("Doctor Access")
        d_id = st.text_input("Doctor ID", value="doc123")
        d_pass = st.text_input("Password", type="password", value="password", key="d_pass")
        if st.button("Log In as Doctor"):
            if d_id and d_pass:
                st.session_state.authenticated = True
                st.session_state.user_role = "Doctor"
                st.rerun()

    st.stop()

# Sidebar & Header
st.sidebar.title(f"Role: {st.session_state.user_role}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.chat_history = []
    st.session_state.last_screening = None
    st.rerun()

st.title(f"🫀 CardioGuard Portal ({st.session_state.user_role} Mode)")

# Diagnostic Screening Form
st.header("1. Clinical Parameters")
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

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca']
patient_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]], columns=feature_names)

if st.button("Run Diagnostic Screening"):
    try:
        model = joblib.load("cardio.plk")
        explainer = joblib.load("sharp_explainer.plk")

        booster_features = model.get_booster().feature_names
        if booster_features:
            for c in booster_features:
                if c not in patient_data.columns:
                    patient_data[c] = 0
            patient_data = patient_data[booster_features]

        risk_prob = model.predict_proba(patient_data)[0][1] * 100
        
        st.session_state.last_screening = {
            "risk_prob": risk_prob,
            "chol": chol,
            "trestbps": trestbps
        }

        st.subheader("Diagnostic Results")
        if risk_prob > 50:
            st.error(f"⚠️ High Heart Attack Risk Detected: **{risk_prob:.1f}%**")
        else:
            st.success(f"✅ Low Heart Attack Risk Detected: **{risk_prob:.1f}%**")

    except Exception as e:
        st.error(f"Error loading prediction model: {e}")

# Consultation Chatbot Section
st.markdown("---")
st.header("2. AI Consultation Assistant")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a consultation question regarding heart health..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    q = prompt.lower()
    screening = st.session_state.last_screening

    if "what to do" in q or "next step" in q:
        if screening and screening["risk_prob"] > 50:
            reply = "🚨 **Recommended Immediate Steps:**\n1. Schedule an immediate consultation with a cardiologist.\n2. Avoid high-stress activities and heavy exertion.\n3. Request an ECG/Echocardiogram."
        else:
            reply = "✅ **Recommended Lifestyle Steps:**\n1. Maintain 30 minutes of daily cardio.\n2. Follow a low-sodium, heart-healthy diet.\n3. Keep routine annual checkups."
    elif "diet" in q or "food" in q:
        reply = "🥗 **Heart Health Diet Tips:**\n- Reduce saturated fats and trans fats.\n- Increase intake of fiber, leafy greens, and omega-3 fatty acids."
    else:
        reply = f"Thank you for reaching out. As a {st.session_state.user_role}, please consult a board-certified physician for personalized medical advice."

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)