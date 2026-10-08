import os
import google.generativeai as genai
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CardioGuard", page_icon="🫀", layout="wide")

# Initialize OpenAI client using Streamlit Secrets or environment variable
client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------------------------------------
# 1. MULTILINGUAL DICTIONARY
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
        "chat_header": "💬 LLM Cardiac Consultation Assistant",
        "chat_placeholder": "Ask anything about risk, diet, medications, or health steps...",
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
        "chat_header": "💬 एलएलएम हृदय परामर्श सहायक",
        "chat_placeholder": "अपने जोखिम, आहार, दवाओं या स्वास्थ्य कदमों के बारे में पूछें...",
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
        "high_risk": "⚠️ ಹೆಚ್ಚಿನ ಹೃದಯಾಘಾತದ ಅಪಾಯ ಕಂಡುಬಂದಿದೆ:",
        "low_risk": "✅ ಕಡಿಮೆ ಹೃದಯಾಘಾತದ ಅಪಾಯ ಕಂಡುಬಂದಿದೆ:",
        "doc_dashboard": "👨‍⚕️ ವೈದ್ಯಕೀಯ ಕ್ಲಿನಿಕಲ್ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "select_patient": "ಪರಿಶೀಲಿಸಲು ರೋಗಿಯನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
        "chat_header": "💬 AI ಸಂವಾದಾತ್ಮಕ ವೈದ್ಯಕೀಯ ಸಮಾಲೋಚನೆ ಸಹಾಯಕ",
        "chat_placeholder": "ನಿಮ್ಮ ಅಪಾಯ, ಆಹಾರ ಪದ್ಧತಿಯ ಬಗ್ಗೆ ಕೇಳಿ...",
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
        "chat_header": "💬 ఏఐ కార్డియాక్ కన్సల్టేషన్ అసిస్టెంట్",
        "chat_placeholder": "మీ ప్రమాదం, ఆహారం గురించి అడగండి...",
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
        "high_risk": "⚠️ மாரடைப்பு ஏற்படும் அபாயம் அதிகம்:",
        "low_risk": "✅ மாரடைப்பு ஏற்படும் அபாயம் குறைவு:",
        "doc_dashboard": "👨‍⚕️ மருத்துவர் மருத்துவ டாஷ்போர்டு",
        "select_patient": "பரிசீலிக்க நோயாளியைத் தேர்ந்தெடுக்கவும்:",
        "chat_header": "💬 AI கார்டியாக் ஆலோசனைக் உதவியாளர்",
        "chat_placeholder": "உங்கள் அபாயம், உணவு குறித்து கேட்கவும்...",
    },
}

# ---------------------------------------------------------
# 2. SESSION STATE & DATABASE SETUP
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
# 3. LOGIN PAGE
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

# Logout Sidebar
st.sidebar.markdown(f"**Role:** {t['role_' + st.session_state.user_role.lower()]}")
if st.sidebar.button(t["logout"]):
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.active_id = "P101"
    st.session_state.chat_history = []
    st.rerun()

st.title(t["title"])

# Safety Check for Active Patient ID
if (
    st.session_state.active_id is None
    or st.session_state.active_id not in st.session_state.database
):
    st.session_state.active_id = list(st.session_state.database.keys())[0]

# ---------------------------------------------------------
# 4. DOCTOR VS PATIENT VIEWS
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

    # ✅ CORRECTED CODE
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