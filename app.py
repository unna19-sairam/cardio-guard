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

        import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="CardioGuard", page_icon="🫀", layout="wide"
)

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_screening" not in st.session_state:
    st.session_state.last_screening = None

# ---------------------------------------------------------
# LOGIN SYSTEM
# ---------------------------------------------------------
if not st.session_state.authenticated:
    st.title("🫀 CardioGuard Portal Login")
    
    tab1, tab2 = st.tabs(["Patient Login", "Doctor Login"])
    
    with tab1:
        st.subheader("Patient Access")
        patient_id = st.text_input("Patient ID / Username", value="patient123")
        patient_pass = st.text_input("Password", type="password", value="password")
        if st.button("Log In as Patient"):
            if patient_id and patient_pass:
                st.session_state.authenticated = True
                st.session_state.user_role = "Patient"
                st.rerun()
            else:
                st.error("Please enter credentials.")

    with tab2:
        st.subheader("Doctor Access")
        doc_id = st.text_input("Doctor ID / License No.", value="doc123")
        doc_pass = st.text_input("Password ", type="password", value="password")
        if st.button("Log In as Doctor"):
            if doc_id and doc_pass:
                st.session_state.authenticated = True
                st.session_state.user_role = "Doctor"
                st.rerun()
            else:
                st.error("Please enter credentials.")

    st.stop()

# ---------------------------------------------------------
# MAIN APP HEADER & LOGOUT
# ---------------------------------------------------------
st.sidebar.title(f"Logged in as: {st.session_state.user_role}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.chat_history = []
    st.session_state.last_screening = None
    st.rerun()

st.title(f"🫀 CardioGuard: Diagnostic & Consultation ({st.session_state.user_role} View)")

# ---------------------------------------------------------
# DIAGNOSTIC FORM
# ---------------------------------------------------------
st.header("1. Patient Clinical Parameters")
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
            "trestbps": trestbps,
            "thalach": thalach
        }

        st.subheader("Diagnostic Results")
        if risk_prob > 50:
            st.error(f"⚠️ High Heart Attack Risk Detected: **{risk_prob:.1f}%**")
        else:
            st.success(f"✅ Low Heart Attack Risk Detected: **{risk_prob:.1f}%**")

    except Exception as e:
        st.error(f"Error executing model diagnostic: {e}")

# ---------------------------------------------------------
# INTERACTIVE CONSULTATION CHATBOT
# ---------------------------------------------------------
st.markdown("---")
st.header("2. AI Medical Consultation Assistant")
st.write("Ask questions about your screening results, dietary suggestions, or next clinical steps.")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask a follow-up question regarding your heart health..."):
    # Append User Message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simple Context-Aware Rule-Based Bot Response
    user_q = prompt.lower()
    screening = st.session_state.last_screening

    if "what to do" in user_q or "next step" in user_q:
        if screening and screening["risk_prob"] > 50:
            reply = "🚨 **Recommended Immediate Actions:**\n1. Schedule a consultation with a cardiologist within 48 hours.\n2. Avoid heavy physical exertion.\n3. Request an ECG and Echocardiogram assessment."
        else:
            reply = "✅ **Maintenance Recommendations:**\n1. Maintain regular 30-minute cardio exercises 5 days a week.\n2. Follow a Mediterranean-style low-sodium diet.\n3. Schedule routine annual blood work."
    elif "diet" in user_q or "food" in user_q:
        reply = "🥗 **Cardiovascular Nutrition Advice:**\n- Reduce saturated fats and refined sugars.\n- Increase intake of soluble fiber (oats, beans, lentils).\n- Incorporate Omega-3 fatty acids (salmon, walnuts, flaxseeds)."
    elif "doctor" in user_q or "appointment" in user_q:
        reply = "👨‍⚕️ Please present your CardioGuard risk probability score and ST-depression parameters directly to your physician during your visit."
    else:
        reply = f"Thank you for your question. As a {st.session_state.user_role}, please consult with your healthcare provider for personalized prescriptions. (Current screening risk: {screening['risk_prob']:.1f}% if available)." if screening else "Please run a Diagnostic Screening above first so I can analyze your results!"

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)