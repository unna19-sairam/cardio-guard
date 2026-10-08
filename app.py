import os
import google.generativeai as genai
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CardioGuard", page_icon="🫀", layout="wide")

# ---------------------------------------------------------
# 1. DYNAMIC LLM ENGINE SETUP (GEMINI API)
# ---------------------------------------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

llm_model = None
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        
        # Dynamically discover active models supporting generateContent
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # Priority order for selecting the best active model
        preferred_models = [
            "models/gemini-2.5-flash",
            "models/gemini-1.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-pro"
        ]
        
        selected_model_name = None
        for p in preferred_models:
            if p in available_models:
                selected_model_name = p
                break
                
        if not selected_model_name and available_models:
            selected_model_name = available_models[0]
            
        if selected_model_name:
            llm_model = genai.GenerativeModel(selected_model_name)
    except Exception as e:
        st.sidebar.warning(f"Gemini API initialization warning: {e}")

# ---------------------------------------------------------
# 2. MULTILINGUAL UI DICTIONARY
# ---------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "🫀 CardioGuard: Multi-Role Cardiac Platform",
        "login_header": "CardioGuard Portal Login",
        "patient_login": "Patient Login",
        "doctor_login": "Doctor Login",
        "logout": "Logout",
        "role_patient": "Patient",
        "role_doctor": "Doctor",
        "run_screening": "Run Diagnostic Screening",
        "high_risk": "⚠️ High Heart Attack Risk Detected:",
        "low_risk": "✅ Low Heart Attack Risk Detected:",
        "doc_dashboard": "👨‍⚕️ Doctor Clinical Dashboard",
        "select_patient": "Select Patient to Review:",
        "chat_header": "💬 Interactive AI Clinical Assistant",
        "chat_placeholder": "Ask about cardiac risk factors, dietary plans, or lifestyle modifications...",
    },
    "Hindi": {
        "title": "🫀 कार्डियोगार्ड: बहु-भूमिका कार्डियक प्लेटफॉर्म",
        "login_header": "कार्डियोगार्ड पोर्टल लॉगिन",
        "patient_login": "मरीज़ लॉगिन",
        "doctor_login": "डॉक्टर लॉगिन",
        "logout": "लॉग आउट",
        "role_patient": "मरीज़",
        "role_doctor": "डॉक्टर",
        "run_screening": "निदान जांच चलाएं",
        "high_risk": "⚠️ हृदयघात का उच्च जोखिम पाया गया:",
        "low_risk": "✅ हृदयघात का कम जोखिम पाया गया:",
        "doc_dashboard": "👨‍⚕️ डॉक्टर क्लिनिकल डैशबोर्ड",
        "select_patient": "समीक्षा के लिए मरीज़ चुनें:",
        "chat_header": "💬 एआई कार्डियक कंसल्टेंट (AI असिस्टेंट)",
        "chat_placeholder": "अपने दिल के स्वास्थ्य, आहार या सलाह के बारे में पूछें...",
    },
    "Kannada": {
        "title": "🫀 ಕಾರ್ಡಿಯೋಗಾರ್ಡ್: ಮಲ್ಟಿ-ರೋಲ್ ಕಾರ್ಡಿಯಾಕ್ ಪ್ಲಾಟ್‌ಫಾರ್ಮ್",
        "login_header": "ಕಾರ್ಡಿಯೋಗಾರ್ಡ್ ಪೋರ್ಟಲ್ ಲಾಗಿನ್",
        "patient_login": "ರೋಗಿಯ ಲಾಗಿನ್",
        "doctor_login": "ವೈದ್ಯರ ಲಾಗಿನ್",
        "logout": "ಲಾಗ್‌ಔಟ್",
        "role_patient": "ರೋಗಿ",
        "role_doctor": "ವೈದ್ಯರು",
        "run_screening": "ರೋಗನಿರ್ಣಯ ಪರೀಕ್ಷೆಯನ್ನು ಚಾಲನೆ ಮಾಡಿ",
        "high_risk": "⚠️ ಹೆಚ್ಚಿನ ಅಪಾಯ ಕಂಡುಬಂದಿದೆ:",
        "low_risk": "✅ ಕಡಿಮೆ ಅಪಾಯ ಕಂಡುಬಂದಿದೆ:",
        "doc_dashboard": "👨‍⚕️ ವೈದ್ಯಕೀಯ ಕ್ಲಿನಿಕಲ್ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "select_patient": "ಸಮೀಕ್ಷೆಗೆ ರೋಗಿಯನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
        "chat_header": "💬 AI ಕಾರ್ಡಿಯಾಕ್ ಸಲಹೆಗಾರ",
        "chat_placeholder": "ನಿಮ್ಮ ಆರೋಗ್ಯ, ಆಹಾರದ ಬಗ್ಗೆ ಕೇಳಿ...",
    },
    "Telugu": {
        "title": "🫀 కార్డియోగార్డ్: మల్టీ-రోల్ కార్డియాక్ ప్లాట్‌ఫారమ్",
        "login_header": "కార్డియోగార్డ్ పోర్టల్ లాగిన్",
        "patient_login": "పేషెంట్ లాగిన్",
        "doctor_login": "డాక్టర్ లాగిన్",
        "logout": "లాగౌట్",
        "role_patient": "పేషెంట్",
        "role_doctor": "డాక్టర్",
        "run_screening": "డయాగ్నోస్టిక్ స్క్రీనింగ్ రన్ చేయండి",
        "high_risk": "⚠️ గుండెపోటు వచ్చే ప్రమాదం ఎక్కువగా ఉంది:",
        "low_risk": "✅ గుండెపోటు వచ్చే ప్రమాదం తక్కువగా ఉంది:",
        "doc_dashboard": "👨‍⚕️ డాక్టర్ క్లినికల్ డాష్‌బోర్డ్",
        "select_patient": "పరిశీలించడానికి పేషెంట్‌ను ఎంచుకోండి:",
        "chat_header": "💬 ఏఐ కార్డియాక్ కన్సల్టెంట్",
        "chat_placeholder": "మీ ఆరోగ్య, ఆహార సమస్యలను అడగండి...",
    },
    "Tamil": {
        "title": "🫀 கார்டியோகார்ட்: மல்டி-ரோல் கார்டியாக் தளம்",
        "login_header": "கார்டியோகார்ட் போர்ட்டல் உள்நுழைவு",
        "patient_login": "நோயாளி உள்நுழைவு",
        "doctor_login": "மருத்துவர் உள்நுழைவு",
        "logout": "வெளியேறு",
        "role_patient": "நோயாளி",
        "role_doctor": "மருத்துவர்",
        "run_screening": "பரிசோதனையை இயக்கவும்",
        "high_risk": "⚠️ அபாயம் அதிகம்:",
        "low_risk": "✅ அபாயம் குறைவு:",
        "doc_dashboard": "👨‍⚕️ மருத்துவர் மருத்துவ டாஷ்போர்டு",
        "select_patient": "நோயாளியைத் தேர்ந்தெடுக்கவும்:",
        "chat_header": "💬 ஏஐ இதய ஆலோசனைக் உதவியாளர்",
        "chat_placeholder": "உங்கள் ஆரோக்கியம் குறித்து கேட்கவும்...",
    },
}

# ---------------------------------------------------------
# 3. SESSION STATE & PATIENT DATABASE
# ---------------------------------------------------------
if "database" not in st.session_state:
    st.session_state.database = {
        "P101": {
            "name": "Rahul Sharma",
            "pass": "pass123",
            "age": 58,
            "sex": 1,
            "cp": 2,
            "trestbps": 145,
            "chol": 260,
            "fbs": 1,
            "restecg": 1,
            "thalach": 130,
            "exang": 1,
            "oldpeak": 2.3,
            "slope": 1,
            "ca": 2,
            "risk": None,
        },
        "P102": {
            "name": "Ananya Roy",
            "pass": "pass123",
            "age": 42,
            "sex": 0,
            "cp": 0,
            "trestbps": 118,
            "chol": 195,
            "fbs": 0,
            "restecg": 0,
            "thalach": 168,
            "exang": 0,
            "oldpeak": 0.2,
            "slope": 2,
            "ca": 0,
            "risk": None,
        },
    }

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "active_id" not in st.session_state:
    st.session_state.active_id = "P101"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Language Selection
st.sidebar.title("🌐 Language / भाषा")
lang = st.sidebar.selectbox(
    "Choose Language:", ["English", "Hindi", "Kannada", "Telugu", "Tamil"]
)
t = TRANSLATIONS[lang]

# ---------------------------------------------------------
# 4. AUTHENTICATION MODULE
# ---------------------------------------------------------
if not st.session_state.authenticated:
    st.title(t["login_header"])
    tab1, tab2 = st.tabs([t["patient_login"], t["doctor_login"]])

    with tab1:
        st.subheader(t["patient_login"])
        p_id = st.text_input("Patient ID (e.g., P101, P102)", value="P101")
        p_pass = st.text_input(
            "Password", type="password", value="pass123", key="p_pass"
        )
        if st.button("Login as Patient"):
            if (
                p_id in st.session_state.database
                and st.session_state.database[p_id]["pass"] == p_pass
            ):
                st.session_state.authenticated = True
                st.session_state.user_role = "Patient"
                st.session_state.active_id = p_id
                st.rerun()
            else:
                st.error("Invalid Patient Credentials")

    with tab2:
        st.subheader(t["doctor_login"])
        d_id = st.text_input("Doctor ID", value="DOC01")
        d_pass = st.text_input(
            "Password", type="password", value="doc123", key="d_pass"
        )
        if st.button("Login as Doctor"):
            if d_id == "DOC01" and d_pass == "doc123":
                st.session_state.authenticated = True
                st.session_state.user_role = "Doctor"
                st.session_state.active_id = "P101"
                st.rerun()
            else:
                st.error("Invalid Doctor Credentials")

    st.stop()

# Sidebar Control
st.sidebar.markdown(f"**Role:** {t['role_' + st.session_state.user_role.lower()]}")
if st.sidebar.button(t["logout"]):
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.active_id = "P101"
    st.session_state.chat_history = []
    st.rerun()

st.title(t["title"])

# State Safety Safeguard
if (
    st.session_state.active_id is None
    or st.session_state.active_id not in st.session_state.database
):
    st.session_state.active_id = list(st.session_state.database.keys())[0]

# ---------------------------------------------------------
# 5. CLINICAL DASHBOARDS
# ---------------------------------------------------------
if st.session_state.user_role == "Doctor":
    st.header(t["doc_dashboard"])

    selected_p = st.selectbox(
        t["select_patient"],
        list(st.session_state.database.keys()),
        index=list(st.session_state.database.keys()).index(
            st.session_state.active_id
        ),
    )
    st.session_state.active_id = selected_p

    records = []
    for pid, data in st.session_state.database.items():
        records.append(
            {
                "Patient ID": pid,
                "Name": data["name"],
                "Age": data["age"],
                "BP (mm Hg)": data["trestbps"],
                "Cholesterol": data["chol"],
                "Calculated Risk": (
                    f"{data['risk']:.1f}%"
                    if data["risk"] is not None
                    else "Not Screened"
                ),
            }
        )
    st.table(pd.DataFrame(records))
    st.markdown("---")

p_data = st.session_state.database[st.session_state.active_id]
st.subheader(f"Patient Profile: {p_data['name']} (ID: {st.session_state.active_id})")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 20, 100, int(p_data["age"]))
    sex = st.selectbox(
        "Sex",
        [0, 1],
        index=int(p_data["sex"]),
        format_func=lambda x: "Female" if x == 0 else "Male",
    )
    cp = st.selectbox(
        "Chest Pain Type (0-3)", [0, 1, 2, 3], index=int(p_data["cp"])
    )
    trestbps = st.number_input(
        "Resting BP (mm Hg)", 80, 200, int(p_data["trestbps"])
    )

with col2:
    chol = st.number_input(
        "Cholesterol (mg/dl)", 100, 600, int(p_data["chol"])
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120", [0, 1], index=int(p_data["fbs"])
    )
    restecg = st.selectbox(
        "Resting ECG", [0, 1, 2], index=int(p_data["restecg"])
    )
    thalach = st.number_input(
        "Max Heart Rate", 60, 220, int(p_data["thalach"])
    )

with col3:
    exang = st.selectbox(
        "Exercise Angina", [0, 1], index=int(p_data["exang"])
    )
    oldpeak = st.number_input(
        "ST Depression", 0.0, 6.2, float(p_data["oldpeak"])
    )
    slope = st.selectbox("ST Slope", [0, 1, 2], index=int(p_data["slope"]))
    ca = st.selectbox(
        "Fluoroscopy Vessels (0-3)", [0, 1, 2, 3], index=int(p_data["ca"])
    )

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
input_df = pd.DataFrame(
    [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]],
    columns=feature_names,
)

if st.button(t["run_screening"]):
    try:
        model = joblib.load("cardio.plk")
        booster_features = model.get_booster().feature_names
        if booster_features:
            for c in booster_features:
                if c not in input_df.columns:
                    input_df[c] = 0
            input_df = input_df[booster_features]

        risk_prob = model.predict_proba(input_df)[0][1] * 100
        p_data["risk"] = risk_prob

        if risk_prob > 50:
            st.error(f"{t['high_risk']} **{risk_prob:.1f}%**")
        else:
            st.success(f"{t['low_risk']} **{risk_prob:.1f}%**")

    except Exception as e:
        st.error(f"Error running diagnostic model: {e}")

# ---------------------------------------------------------
# 6. DYNAMIC LLM-POWERED CHATBOT
# ---------------------------------------------------------
st.markdown("---")
st.header(t["chat_header"])

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(t["chat_placeholder"]):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    risk_val = (
        f"{p_data['risk']:.1f}%" if p_data["risk"] is not None else "Not Screened"
    )

    system_instructions = f"""
    You are an expert AI Cardiac Clinical Assistant inside CardioGuard platform.
    
    PATIENT PROFILE:
    - Name: {p_data['name']}
    - Age: {age}, Sex: {'Male' if sex==1 else 'Female'}
    - BP: {trestbps} mm Hg
    - Cholesterol: {chol} mg/dl
    - Fasting Sugar > 120: {'Yes' if fbs==1 else 'No'}
    - Max Heart Rate: {thalach} bpm
    - ST Depression: {oldpeak}
    - Calculated Risk Score: {risk_val}
    
    INSTRUCTIONS:
    1. Respond STRICTLY in language: {lang}.
    2. Format using clear Markdown formatting (bullet points, bold highlights).
    3. Refer directly to patient metrics when applicable.
    """

    full_prompt = f"{system_instructions}\n\nUSER QUESTION: {prompt}"

    with st.chat_message("assistant"):
        with st.spinner("Analyzing patient clinical metrics..."):
            if llm_model:
                try:
                    res = llm_model.generate_content(full_prompt)
                    reply = res.text
                except Exception as err:
                    reply = f"⚠️ API Error ({type(err).__name__}): {str(err)}"
            else:
                reply = (
                    f"**[Demo Mode — Add GEMINI_API_KEY to Secrets]**\n\n"
                    f"Summary for {p_data['name']}:\n"
                    f"- BP: {trestbps} mm Hg | Cholesterol: {chol} mg/dl | Risk: {risk_val}"
                )

            st.markdown(reply)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply}
            )