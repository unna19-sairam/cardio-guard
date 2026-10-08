import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CardioGuard", page_icon="🫀", layout="wide"
)

# ---------------------------------------------------------
# 1. MULTILINGUAL DICTIONARY
# ---------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "🫀 CardioGuard: Multi-Role Cardiac Platform",
        "login_header": "CardioGuard Portal Login",
        "select_lang": "Select Language",
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
        "patient_records": "All Patient Records",
        "chat_header": "💬 Interactive Cardiac Consultation Assistant",
        "chat_placeholder": "Ask about your risk, diet, or clinical next steps...",
        "risk_summary": "Patient Risk Analysis:",
    },
    "Hindi": {
        "title": "🫀 कार्डियोगार्ड: बहु-भूमिका कार्डियक प्लेटफॉर्म",
        "login_header": "कार्डियोगार्ड पोर्टल लॉगिन",
        "select_lang": "भाषा चुनें",
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
        "patient_records": "सभी मरीजों के रिकॉर्ड",
        "chat_header": "💬 इंटरएक्टिव हृदय परामर्श सहायक",
        "chat_placeholder": "अपने जोखिम, आहार या अगले चरणों के बारे में पूछें...",
        "risk_summary": "मरीज़ के जोखिम का विश्लेषण:",
    },
    "Kannada": {
        "title": "🫀 ಕಾರ್ಡಿಯೋಗಾರ್ಡ್: ಮಲ್ಟಿ-ರೋಲ್ ಕಾರ್ಡಿಯಾಕ್ ಪ್ಲಾಟ್‌ಫಾರ್ಮ್",
        "login_header": "ಕಾರ್ಡಿಯೋಗಾರ್ಡ್ ಪೋರ್ಟಲ್ ಲಾಗಿನ್",
        "select_lang": "ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
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
        "patient_records": "ಎಲ್ಲಾ ರೋಗಿಗಳ ದಾಖಲೆಗಳು",
        "chat_header": "💬 ಸಂವಾದಾತ್ಮಕ ಹೃದಯ ಸಮಾಲೋಚನೆ ಸಹಾಯಕ",
        "chat_placeholder": "ನಿಮ್ಮ ಅಪಾಯ, ಆಹಾರ ಪದ್ಧತಿಯ ಬಗ್ಗೆ ಕೇಳಿ...",
        "risk_summary": "ರೋಗಿಯ ಅಪಾಯದ ವಿಶ್ಲೇಷಣೆ:",
    },
    "Telugu": {
        "title": "🫀 కార్డియోగార్డ్: మల్టీ-రోల్ కార్డియాక్ ప్లాట్‌ఫారమ్",
        "login_header": "కార్డియోగార్డ్ పోర్టల్ లాగిన్",
        "select_lang": "భాషను ఎంచుకోండి",
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
        "patient_records": "అన్ని పేషెంట్ రికార్డులు",
        "chat_header": "💬 ఇంటరాక్టివ్ కార్డియాక్ కన్సల్టేషన్ అసిస్టెంట్",
        "chat_placeholder": "మీ ప్రమాదం, ఆహారం గురించి అడగండి...",
        "risk_summary": "పేషెంట్ రిస్క్ విశ్లేషణ:",
    },
    "Tamil": {
        "title": "🫀 கார்டியோகார்ட்: மல்டி-ரோல் கார்டியாக் தளம்",
        "login_header": "கார்டியோகார்ட் போர்ட்டல் உள்நுழைவு",
        "select_lang": "மொழியைத் தேர்ந்தெடுக்கவும்",
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
        "patient_records": "அனைத்து நோயாளி பதிவுகள்",
        "chat_header": "💬 கார்டியாக் ஆலோசனைக் உதவியாளர்",
        "chat_placeholder": "உங்கள் அபாயம், உணவு குறித்து கேட்கவும்...",
        "risk_summary": "நோயாளி அபாய பகுப்பாய்வு:",
    },
}

# ---------------------------------------------------------
# 2. MOCK PATIENT DATABASE
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
    st.session_state.active_id = None
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
    st.session_state.active_id = None
    st.session_state.chat_history = []
    st.rerun()

st.title(t["title"])

# ---------------------------------------------------------
# 4. DOCTOR VS PATIENT VIEWS
# ---------------------------------------------------------
if st.session_state.user_role == "Doctor":
    st.header(t["doc_dashboard"])

    # Doctor patient-switcher dropdown
    selected_p = st.selectbox(
        t["select_patient"], list(st.session_state.database.keys())
    )
    st.session_state.active_id = selected_p

    # Doctor Table view of all patients
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

# Form Parameters
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
        st.error(f"Error executing prediction: {e}")

# ---------------------------------------------------------
# 5. DYNAMIC MULTILINGUAL CONSULTATION BOT
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

    # Dynamic Analysis based on specific patient metrics
    risk_val = p_data["risk"] if p_data["risk"] is not None else 0.0
    q = prompt.lower()

    if lang == "English":
        if risk_val > 50:
            reply = f"🚨 **High Risk Analysis ({risk_val:.1f}%):**\n- BP ({trestbps} mmHg) and Cholesterol ({chol} mg/dl) are key drivers.\n- **Action:** Schedule a clinical cardiologist checkup within 48 hours."
        elif "diet" in q or "food" in q:
            reply = f"🥗 **Personalized Diet Plan for {p_data['name']}:**\n- Since cholesterol is {chol} mg/dl, restrict saturated fats.\n- Limit sodium intake below 2,000 mg/day for BP control."
        else:
            reply = f"✅ **Low Risk Maintenance ({risk_val:.1f}%):** Keep up regular 30-minute daily cardio exercises and routine annual screenings."

    elif lang == "Hindi":
        if risk_val > 50:
            reply = f"🚨 **उच्च जोखिम विश्लेषण ({risk_val:.1f}%):**\n- आपका रक्तचाप ({trestbps} mmHg) और कोलेस्ट्रॉल ({chol} mg/dl) मुख्य कारण हैं।\n- **कार्रवाई:** 48 घंटे के भीतर कार्डियोलॉजिस्ट से संपर्क करें।"
        else:
            reply = f"✅ **कम जोखिम स्थिति ({risk_val:.1f}%):** रोजाना 30 मिनट व्यायाम करें और संतुलित आहार बनाए रखें।"

    elif lang == "Kannada":
        if risk_val > 50:
            reply = f"🚨 **ಹೆಚ್ಚಿನ ಅಪಾಯದ ವಿಶ್ಲೇಷಣೆ ({risk_val:.1f}%):**\n- ನಿಮ್ಮ ರಕ್ತದೊತ್ತಡ ({trestbps} mmHg) ಮತ್ತು ಕೊಲೆಸ್ಟ್ರಾಲ್ ({chol} mg/dl) ಪ್ರಮುಖ ಕಾರಣಗಳಾಗಿವೆ.\n- **ಸಲಹೆ:** 48 ಗಂಟೆಗಳ ಒಳಗೆ ಹೃದ್ರೋಗ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
        else:
            reply = f"✅ **ಕಡಿಮೆ ಅಪಾಯದ ಸ್ಥಿತಿ ({risk_val:.1f}%):** ಪ್ರತಿದಿನ 30 ನಿಮಿಷಗಳ ಕಾಲ ವ್ಯಾಯಾಮ ಮಾಡಿ."

    elif lang == "Telugu":
        if risk_val > 50:
            reply = f"🚨 **అధిక రిస్క్ విశ్లేషణ ({risk_val:.1f}%):**\n- మీ బీపీ ({trestbps} mmHg) మరియు కొలెస్ట్రాల్ ({chol} mg/dl) ప్రధాన కారణాలు.\n- **చర్య:** 48 గంటల్లో కార్డియాలజిస్ట్‌ను సంప్రదించండి."
        else:
            reply = f"✅ **తక్కువ రిస్క్ పరిస్థితి ({risk_val:.1f}%):** ప్రతిరోజూ 30 నిమిషాలు వ్యాయామం చేయండి."

    elif lang == "Tamil":
        if risk_val > 50:
            reply = f"🚨 **அதிக அபாய பகுப்பாய்வு ({risk_val:.1f}%):**\n- உங்கள் ரத்த அழுத்தம் ({trestbps} mmHg) மற்றும் கொலஸ்ட்ரால் ({chol} mg/dl) முக்கிய காரணங்கள்.\n- **நடவடிக்கை:** 48 மணி நேரத்திற்குள் மருத்துவரை அணுகவும்."
        else:
            reply = f"✅ **குறைந்த அபாய நிலை ({risk_val:.1f}%):** தினமும் 30 நிமிடங்கள் உடற்பயிற்சி செய்யுங்கள்."

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)