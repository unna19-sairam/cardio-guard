import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Try importing Google GenAI SDK (new and fallback legacy)
try:
    from google import genai
    GENAI_NEW_AVAILABLE = True
except ImportError:
    GENAI_NEW_AVAILABLE = False

try:
    import google.generativeai as genai_legacy
    GENAI_LEGACY_AVAILABLE = True
except ImportError:
    GENAI_LEGACY_AVAILABLE = False

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioGuard - AI Cardiac Risk & Explanation",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 6px;
    }
    .high-risk-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: bold;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #FCA5A5;
    }
    .mod-risk-badge {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: bold;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #FCD34D;
    }
    .low-risk-badge {
        background-color: #D1FAE5;
        color: #065F46;
        font-weight: bold;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #6EE7B7;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. MULTILINGUAL UI DICTIONARY (7 LANGUAGES)
# ---------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "🫀 CardioGuard: Multi-Role Cardiac Risk & Cause Platform",
        "subtitle": "Dual ML Models (Risk Detector + Cause Explainer) & Interactive AI LLM",
        "login_header": "CardioGuard Portal Login",
        "patient_login": "Patient Login",
        "doctor_login": "Doctor Login",
        "patient_id": "Patient ID (e.g., P101, P102)",
        "doctor_id": "Doctor ID (e.g., DOC01)",
        "password": "Password",
        "login_patient_btn": "Login as Patient",
        "login_doctor_btn": "Login as Doctor",
        "logout": "Logout",
        "role_patient": "Patient",
        "role_doctor": "Doctor",
        "patient_dash": "👤 Patient Personal Health Dashboard",
        "doctor_dash": "👨‍⚕️ Doctor Clinical Command Dashboard",
        "my_records": "📋 My Health Profile & Enter Vitals",
        "update_vitals": "Save & Update Vitals",
        "all_patients": "📊 Patients Directory Overview",
        "select_patient": "Select Patient to Review & Manage:",
        "add_patient": "➕ Register New Patient Record",
        "new_p_id": "New Patient ID",
        "new_p_name": "Full Name",
        "new_p_pass": "Account Password",
        "create_p_btn": "Register Patient Record",
        "model1_header": "🧪 Model #1: Heart Attack Risk Detection",
        "model2_header": "🔍 Model #2: Heart Attack Cause Explainer (SHAP Analysis)",
        "run_screening": "Run Diagnostic Risk Screening",
        "run_explanation": "Analyze Causes of Heart Attack Risk",
        "risk_score": "Calculated Heart Attack Risk Score",
        "high_risk": "⚠️ HIGH HEART ATTACK RISK",
        "mod_risk": "⚠️ MODERATE HEART ATTACK RISK",
        "low_risk": "✅ LOW HEART ATTACK RISK",
        "top_causes": "🚨 Primary Contributing Causes (Pushing Risk UP)",
        "protective": "🛡️ Protective / Low-Risk Factors (Pulling Risk DOWN)",
        "shap_chart": "📊 ML Feature Attribution (Cause Breakdown)",
        "chat_header": "💬 Interactive AI Clinical Assistant (LLM)",
        "chat_placeholder": "Ask about heart risk factors, cause analysis, diet, or treatment plans...",
        "api_key_prompt": "Gemini API Key (Optional override):",
        "vitals_saved": "✅ Patient health vitals updated successfully!",
        "invalid_creds": "❌ Invalid login credentials. Please check your ID and password.",
    },
    "Hindi": {
        "title": "🫀 कार्डियोगार्ड: बहु-भूमिका कार्डियक जोखिम एवं कारण विश्लेषण प्लेटफॉर्म",
        "subtitle": "दोहरी एमएल मॉडल (जोखिम पहचान + कारण स्पष्टीकरण) एवं इंटरैक्टिव एआई एलएलएम",
        "login_header": "कार्डियोगार्ड पोर्टल लॉगिन",
        "patient_login": "मरीज़ लॉगिन",
        "doctor_login": "डॉक्टर लॉगिन",
        "patient_id": "मरीज़ आईडी (उदा. P101, P102)",
        "doctor_id": "डॉक्टर आईडी (उदा. DOC01)",
        "password": "पासवर्ड",
        "login_patient_btn": "मरीज़ के रूप में लॉगिन करें",
        "login_doctor_btn": "डॉक्टर के रूप में लॉगिन करें",
        "logout": "लॉग आउट",
        "role_patient": "मरीज़",
        "role_doctor": "डॉक्टर",
        "patient_dash": "👤 मरीज़ व्यक्तिगत स्वास्थ्य डैशबोर्ड",
        "doctor_dash": "👨‍⚕️ डॉक्टर क्लिनिकल डैशबोर्ड",
        "my_records": "📋 मेरा स्वास्थ्य प्रोफाइल और विटल्स दर्ज करें",
        "update_vitals": "विटल्स सहेजें और अपडेट करें",
        "all_patients": "📊 सभी मरीजों की सूची एवं विवरण",
        "select_patient": "समीक्षा के लिए मरीज़ चुनें:",
        "add_patient": "➕ नया मरीज़ रिकॉर्ड जोड़ें",
        "new_p_id": "नया मरीज़ आईडी",
        "new_p_name": "पूरा नाम",
        "new_p_pass": "खाता पासवर्ड",
        "create_p_btn": "मरीज़ रिकॉर्ड पंजीकृत करें",
        "model1_header": "🧪 मॉडल #1: हृदयघात का जोखिम स्तर",
        "model2_header": "🔍 मॉडल #2: हृदयघात के कारणों का विश्लेषण (SHAP एक्सप्लेनर)",
        "run_screening": "जोखिम जांच चलाएं",
        "run_explanation": "हृदयघात के कारणों का विश्लेषण करें",
        "risk_score": "अनुमानित हृदयघात जोखिम अंक",
        "high_risk": "⚠️ हृदयघात का उच्च जोखिम",
        "mod_risk": "⚠️ हृदयघात का मध्यम जोखिम",
        "low_risk": "✅ हृदयघात का कम जोखिम",
        "top_causes": "🚨 मुख्य जोखिम कारक (जोखिम बढ़ाने वाले कारण)",
        "protective": "🛡️ सुरक्षात्मक कारक (जोखिम घटाने वाले कारण)",
        "shap_chart": "📊 एमएल फीचर योगदान चार्ट (कारणों का विवरण)",
        "chat_header": "💬 इंटरैक्टिव एआई क्लिनिकल असिस्टेंट (LLM)",
        "chat_placeholder": "हृदय स्वास्थ्य, आहार या सलाह के बारे में पूछें...",
        "api_key_prompt": "जेमिनी एपीआई कुंजी (वैकल्पिक):",
        "vitals_saved": "✅ मरीज़ के विटल्स सफलतापूर्वक अपडेट किए गए!",
        "invalid_creds": "❌ अमान्य क्रेडेंशियल। कृपया पुनः प्रयास करें।",
    },
    "Kannada": {
        "title": "🫀 ಕಾರ್ಕಾರ್ಡಿಯೋಗಾರ್ಡ್: ಮಲ್ಟಿ-ರೋಲ್ ಕಾರ್ಡಿಯಾಕ್ ಪ್ಲಾಟ್‌ಫಾರ್ಮ್",
        "subtitle": "ದ್ವಿ ML ಮಾದರಿಗಳು (ಅಪಾಯ ಪತ್ತೆ + ಕಾರಣ ವಿಶ್ಲೇಷಣೆ) ಮತ್ತು ಸಂವಾದಾತ್ಮಕ AI LLM",
        "login_header": "ಕಾರ್ಡಿಯೋಗಾರ್ಡ್ ಪೋರ್ಟಲ್ ಲಾಗಿನ್",
        "patient_login": "ರೋಗಿಯ ಲಾಗಿನ್",
        "doctor_login": "ವೈದ್ಯರ ಲಾಗಿನ್",
        "patient_id": "ರೋಗಿ ID (ಉದಾ, P101, P102)",
        "doctor_id": "ವೈದ್ಯರ ID (ಉದಾ, DOC01)",
        "password": "ಪಾಸ್‌ವರ್ಡ್",
        "login_patient_btn": "ರೋಗಿಯಾಗಿ ಲಾಗಿನ್ ಮಾಡಿ",
        "login_doctor_btn": "ವೈದ್ಯರಾಗಿ ಲಾಗಿನ್ ಮಾಡಿ",
        "logout": "ಲಾಗ್‌ಔಟ್",
        "role_patient": "ರೋಗಿ",
        "role_doctor": "ವೈದ್ಯರು",
        "patient_dash": "👤 ರೋಗಿಯ ವೈಯಕ್ತಿಕ ಆರೋಗ್ಯ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "doctor_dash": "👨‍⚕️ ವೈದ್ಯಕೀಯ ಕ್ಲಿನಿಕಲ್ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "my_records": "📋 ನನ್ನ ಆರೋಗ್ಯ ದಾಖಲೆಗಳು ಮತ್ತು ಮಾಹಿತಿಯನ್ನು ನಮೂದಿಸಿ",
        "update_vitals": "ದಾಖಲೆಗಳನ್ನು ಉಳಿಸಿ ಮತ್ತು ನವೀಕರಿಸಿ",
        "all_patients": "📊 ರೋಗಿಗಳ ವಿವರಗಳ ಪಟ್ಟಿ",
        "select_patient": "ಪರಿಶೀಲಿಸಲು ರೋಗಿಯನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
        "add_patient": "➕ ಹೊಸ ರೋಗಿಯ ದಾಖಲೆಯನ್ನು ಸೇರಿಸಿ",
        "new_p_id": "ಹೊಸ ರೋಗಿ ID",
        "new_p_name": "ಪೂರ್ಣ ಹೆಸರು",
        "new_p_pass": "ಖಾತೆ ಪಾಸ್‌ವರ್ಡ್",
        "create_p_btn": "ರೋಗಿಯನ್ನು ನೋಂದಾಯಿಸಿ",
        "model1_header": "🧪 ಮಾದರಿ #1: ಹೃದಯಾಘಾತದ ಅಪಾಯದ ಮಟ್ಟ ಪತ್ತೆ",
        "model2_header": "🔍 ಮಾದರಿ #2: ಹೃದಯಾಘಾತದ ಕಾರಣಗಳ ವಿಶ್ಲೇಷಣೆ (SHAP ಮಾದರಿ)",
        "run_screening": "ಅಪಾಯದ ಪರೀಕ್ಷೆಯನ್ನು ಚಾಲನೆ ಮಾಡಿ",
        "run_explanation": "ಹೃದಯಾಘಾತದ ಕಾರಣಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "risk_score": "ಲೆಕ್ಕಹಾಕಿದ ಅಪಾಯದ ಅಂಕ",
        "high_risk": "⚠️ ಹೆಚ್ಚಿನ ಅಪಾಯ ಕಂಡುಬಂದಿದೆ",
        "mod_risk": "⚠️ ಮಧ್ಯಮ ಅಪಾಯ ಕಂಡುಬಂದಿದೆ",
        "low_risk": "✅ ಕಡಿಮೆ ಅಪಾಯ ಕಂಡುಬಂದಿದೆ",
        "top_causes": "🚨 ಮುಖ್ಯ ಅಪಾಯದ ಕಾರಣಗಳು",
        "protective": "🛡️ ರಕ್ಷಣಾತ್ಮಕ ಅಂಶಗಳು",
        "shap_chart": "📊 ML ವಿವರಣೆ ಚಾರ್ಟ್",
        "chat_header": "💬 ಸಂವಾದಾತ್ಮಕ AI ಸಹಾಯಕ",
        "chat_placeholder": "ನಿಮ್ಮ ಆರೋಗ್ಯ, ಆಹಾರದ ಬಗ್ಗೆ ಕೇಳಿ...",
        "api_key_prompt": "Gemini API Key (ಐಚ್ಛಿಕ):",
        "vitals_saved": "✅ ರೋಗಿಯ ಮಾಹಿತಿಯನ್ನು ಯಶಸ್ವಿಯಾಗಿ ನವೀಕರಿಸಲಾಗಿದೆ!",
        "invalid_creds": "❌ ತಪ್ಪಾದ ಲಾಗಿನ್ ಮಾಹಿತಿ.",
    },
    "Telugu": {
        "title": "🫀 కార్డియోగార్డ్: మల్టీ-రోల్ కార్డియాక్ రిస్క్ & కాజ్ ప్లాట్‌ఫారమ్",
        "subtitle": "ద్వంద్వ ML నమూనాలు (రిస్క్ డిటెక్టర్ + కారణాల విశ్లేషణ) మరియు AI LLM",
        "login_header": "కార్డియోగార్డ్ పోర్టల్ లాగిన్",
        "patient_login": "పేషెంట్ లాగిన్",
        "doctor_login": "డాక్టర్ లాగిన్",
        "patient_id": "పేషెంట్ ID (ఉదా, P101, P102)",
        "doctor_id": "డాక్టర్ ID (ఉదా, DOC01)",
        "password": "పాస్‌వర్డ్",
        "login_patient_btn": "పేషెంట్‌గా లాగిన్ చేయండి",
        "login_doctor_btn": "డాక్టర్‌గా లాగిన్ చేయండి",
        "logout": "లాగౌట్",
        "role_patient": "పేషెంట్",
        "role_doctor": "డాక్టర్",
        "patient_dash": "👤 పేషెంట్ పర్సనల్ హెల్త్ డాష్‌బోర్డ్",
        "doctor_dash": "👨‍⚕️ డాక్టర్ క్లినికల్ డాష్‌బోర్డ్",
        "my_records": "📋 నా ఆరోగ్య ప్రొఫైల్ మరియు వివరాలను నమోదు చేయండి",
        "update_vitals": "వివరాలను సేవ్ చేయండి",
        "all_patients": "📊 రోగులందరి వివరాల జాబితా",
        "select_patient": "పరిశీలించడానికి పేషెంట్‌ను ఎంచుకోండి:",
        "add_patient": "➕ కొత్త పేషెంట్ రికార్డ్‌ను జోడించండి",
        "new_p_id": "కొత్త పేషెంట్ ID",
        "new_p_name": "పూర్తి పేరు",
        "new_p_pass": "ఖాతా పాస్‌వర్డ్",
        "create_p_btn": "పేషెంట్‌ను నమోదు చేయండి",
        "model1_header": "🧪 మోడల్ #1: గుండెపోటు వచ్చే ప్రమాద స్థాయి",
        "model2_header": "🔍 మోడల్ #2: గుండెపోటు కారణాల విశ్లేషణ (SHAP మోడల్)",
        "run_screening": "రిస్క్ స్క్రీనింగ్ రన్ చేయండి",
        "run_explanation": "గుండెపోటు కారణాలను విశ్లేషించండి",
        "risk_score": "లెక్కించిన రిస్క్ స్కోర్",
        "high_risk": "⚠️ గుండెపోటు వచ్చే ప్రమాదం ఎక్కువగా ఉంది",
        "mod_risk": "⚠️ గుండెపోటు వచ్చే ప్రమాదం మధ్యస్థంగా ఉంది",
        "low_risk": "✅ గుండెపోటు వచ్చే ప్రమాదం తక్కువగా ఉంది",
        "top_causes": "🚨 ప్రధాన ప్రమాద కారణాలు",
        "protective": "🛡️ రక్షణ కారకాలు",
        "shap_chart": "📊 ML కారణాల విశ్లేషణ చార్ట్",
        "chat_header": "💬 ఏఐ కార్డియాక్ కన్సల్టెంట్ (LLM)",
        "chat_placeholder": "మీ ఆరోగ్య సమస్యల గురించి అడగండి...",
        "api_key_prompt": "Gemini API Key (ఐచ్ఛికం):",
        "vitals_saved": "✅ పేషెంట్ వివరాలు నవీకరించబడ్డాయి!",
        "invalid_creds": "❌ చెల్లని లాగిన్ వివరాలు.",
    },
    "Tamil": {
        "title": "🫀 கார்டியோகார்ட்: மல்டி-ரோல் கார்டியாக் தளம்",
        "subtitle": "இரட்டை ML மாதிரிகள் (அபாயக் கண்டறிதல் + காரண பகுப்பாய்வு) & AI LLM",
        "login_header": "கார்டியோகார்ட் போர்ட்டல் உள்நுழைவு",
        "patient_login": "நோயாளி உள்நுழைவு",
        "doctor_login": "மருத்துவர் உள்நுழைவு",
        "patient_id": "நோயாளி ID (எ.கா, P101, P102)",
        "doctor_id": "மருத்துவர் ID (எ.கா, DOC01)",
        "password": "கடவுச்சொல்",
        "login_patient_btn": "நோயாளியாக உள்நுழையவும்",
        "login_doctor_btn": "மருத்துவராக உள்நுழையவும்",
        "logout": "வெளியேறு",
        "role_patient": "நோயாளி",
        "role_doctor": "மருத்துவர்",
        "patient_dash": "👤 நோயாளி சுய சுகாதார டாஷ்போர்டு",
        "doctor_dash": "👨‍⚕️ மருத்துவர் மருத்துவ டாஷ்போர்டு",
        "my_records": "📋 என் சுகாதார விவரங்களை உள்ளிடவும்",
        "update_vitals": "விவரங்களைச் சேமிக்கவும்",
        "all_patients": "📊 நோயாளிகள் விவரப் பட்டியல்",
        "select_patient": "நோயாளியைத் தேர்ந்தெடுக்கவும்:",
        "add_patient": "➕ புதிய நோயாளியைப் பதிவுசெய்க",
        "new_p_id": "புதிய நோயாளி ID",
        "new_p_name": "முழு பெயர்",
        "new_p_pass": "கடவுச்சொல்",
        "create_p_btn": "நோயாளியைப் பதிவுசெய்க",
        "model1_header": "🧪 மாதிரி #1: மாரடைப்பு அபாய நிலை கண்டறிதல்",
        "model2_header": "🔍 மாதிரி #2: மாரடைப்பு காரணங்களின் பகுப்பாய்வு (SHAP)",
        "run_screening": "அபாய பரிசோதனையை இயக்கவும்",
        "run_explanation": "காரணங்களை ஆராயவும்",
        "risk_score": "கணக்கிடப்பட்ட அபாய மதிப்பெண்",
        "high_risk": "⚠️ மாரடைப்பு அபாயம் அதிகம்",
        "mod_risk": "⚠️ மாரடைப்பு அபாயம் மிதமானது",
        "low_risk": "✅ மாரடைப்பு அபாயம் குறைவு",
        "top_causes": "🚨 முதன்மை ஆபத்து காரணிகள்",
        "protective": "🛡️ பாதுகாப்பு காரணிகள்",
        "shap_chart": "📊 ML காரணிகள் வரைபடம்",
        "chat_header": "💬 ஏஐ இதய ஆலோசனைக் உதவியாளர் (LLM)",
        "chat_placeholder": "உங்கள் ஆரோக்கியம் குறித்து கேட்கவும்...",
        "api_key_prompt": "Gemini API Key (விருப்பமானது):",
        "vitals_saved": "✅ நோயாளி விவரங்கள் புதுப்பிக்கப்பட்டன!",
        "invalid_creds": "❌ தவறான உள்நுழைவு விவரங்கள்.",
    },
    "Marathi": {
        "title": "🫀 कार्डियोगार्ड: मल्टी-रोल कार्डियाक प्लॅटफॉर्म",
        "subtitle": "दुहेरी एमएल मॉडेल्स (जोखीम शोधणे + कारण स्पष्टीकरण) आणि AI LLM",
        "login_header": "कार्डियोगार्ड पोर्टल लॉगिन",
        "patient_login": "रुग्ण लॉगिन",
        "doctor_login": "डॉक्टर लॉगिन",
        "patient_id": "रुग्ण ID (उदा. P101, P102)",
        "doctor_id": "डॉक्टर ID (उदा. DOC01)",
        "password": "पासवर्ड",
        "login_patient_btn": "रुग्ण म्हणून लॉगिन करा",
        "login_doctor_btn": "डॉक्टर म्हणून लॉगिन करा",
        "logout": "लॉग आउट",
        "role_patient": "रुग्ण",
        "role_doctor": "डॉक्टर",
        "patient_dash": "👤 रुग्ण वैयक्तिक आरोग्य डॅशबोर्ड",
        "doctor_dash": "👨‍⚕️ डॉक्टर क्लिनिकल डॅशबोर्ड",
        "my_records": "📋 माझे आरोग्य प्रोफाइल आणि माहिती नोंदवा",
        "update_vitals": "माहिती जतन करा",
        "all_patients": "📊 सर्व रुग्णांची यादी",
        "select_patient": "रुग्ण निवडा:",
        "add_patient": "➕ नवीन रुग्ण नोंदवा",
        "new_p_id": "नवीन रुग्ण ID",
        "new_p_name": "पूर्ण नाव",
        "new_p_pass": "पासवर्ड",
        "create_p_btn": "रुग्ण नोंदणी करा",
        "model1_header": "🧪 मॉडेल #1: ह्दयविकाराचा झटका जोखीम पातळी",
        "model2_header": "🔍 मॉडेल #2: ह्दयविकाराच्या झटक्याची कारणे (SHAP)",
        "run_screening": "जोखीम चाचणी करा",
        "run_explanation": "कारणांचे विश्लेषण करा",
        "risk_score": "जोखीम गुणोत्तर",
        "high_risk": "⚠️ ह्दयविकाराचा धोका जास्त आहे",
        "mod_risk": "⚠️ ह्दयविकाराचा धोका मध्यम आहे",
        "low_risk": "✅ ह्दयविकाराचा धोका कमी आहे",
        "top_causes": "🚨 मुख्य जोखीम कारणे",
        "protective": "🛡️ संरक्षणात्मक घटक",
        "shap_chart": "📊 ML कारणे आलेख",
        "chat_header": "💬 एआय क्लिनिकल सहाय्यक (LLM)",
        "chat_placeholder": "हृदय आरोग्याबद्दल विचारा...",
        "api_key_prompt": "Gemini API Key (पर्यायी):",
        "vitals_saved": "✅ माहिती यशस्वीरित्या अद्ययावत केली!",
        "invalid_creds": "❌ चुकीचा आयडी किंवा पासवर्ड.",
    },
    "Bengali": {
        "title": "🫀 কার্ডিওগার্ড: মাল্টি-রোল কার্ডিয়াক প্ল্যাটফর্ম",
        "subtitle": "দ্বৈত এমএল মডেল (ঝুঁকি সনাক্তকরণ + কারণ বিশ্লেষণ) এবং AI LLM",
        "login_header": "কার্ডিওগার্ড পোর্টাল লগইন",
        "patient_login": "রোগী লগইন",
        "doctor_login": "ডাক্তার লগইন",
        "patient_id": "রোগী আইডি (যেমন P101, P102)",
        "doctor_id": "ডাক্তার আইডি (যেমন DOC01)",
        "password": "পাসওয়ার্ড",
        "login_patient_btn": "রোগী হিসেবে লগইন করুন",
        "login_doctor_btn": "ডাক্তার হিসেবে লগইন করুন",
        "logout": "লগ আউট",
        "role_patient": "রোগী",
        "role_doctor": "ডাক্তার",
        "patient_dash": "👤 রোগীর ব্যক্তিগত স্বাস্থ্য ড্যাশবোর্ড",
        "doctor_dash": "👨‍⚕️ ডাক্তার ক্লিনিক্যাল ড্যাশবোর্ড",
        "my_records": "📋 আমার স্বাস্থ্য প্রোফাইল এবং তথ্য প্রদান করুন",
        "update_vitals": "তথ্য সংরক্ষণ করুন",
        "all_patients": "📊 সকল রোগীর তালিকা",
        "select_patient": "রোগী নির্বাচন করুন:",
        "add_patient": "➕ নতুন রোগী নিবন্ধন করুন",
        "new_p_id": "নতুন রোগী আইডি",
        "new_p_name": "পূর্ণ নাম",
        "new_p_pass": "পাসওয়ার্ড",
        "create_p_btn": "রোগী নিবন্ধন করুন",
        "model1_header": "🧪 মডেল #১: হার্ট অ্যাটাকের ঝুঁকির মাত্রা",
        "model2_header": "🔍 মডেল #২: হার্ট অ্যাটাকের কারণ বিশ্লেষণ (SHAP)",
        "run_screening": "ঝুঁকি পরীক্ষা চালান",
        "run_explanation": "কারণসমূহ বিশ্লেষণ করুন",
        "risk_score": "ঝুঁকির পরিমাণ",
        "high_risk": "⚠️ হার্ট অ্যাটাকের উচ্চ ঝুঁকি",
        "mod_risk": "⚠️ হার্ট অ্যাটাকের মাঝারি ঝুঁকি",
        "low_risk": "✅ হার্ট অ্যাটাকের কম ঝুঁকি",
        "top_causes": "🚨 প্রধান ঝুঁকির কারণসমূহ",
        "protective": "🛡️ সুরক্ষামূলক উপাদান",
        "shap_chart": "📊 এমএল কারণ বিশ্লেষণ চার্ট",
        "chat_header": "💬 এআই ক্লিনিক্যাল অ্যাসিস্ট্যান্ট (LLM)",
        "chat_placeholder": "হার্টের স্বাস্থ্য সম্পর্কে জিজ্ঞাসা করুন...",
        "api_key_prompt": "Gemini API Key (ঐচ্ছিক):",
        "vitals_saved": "✅ রোগীর তথ্য সফলভাবে আপডেট হয়েছে!",
        "invalid_creds": "❌ ভুল আইডি বা পাসওয়ার্ড।",
    }
}

# Feature human names mapping per feature
FEATURE_LABELS = {
    "age": {"English": "Age (Years)", "Hindi": "आयु (वर्ष)", "Kannada": "ವಯಸ್ಸು", "Telugu": "వయస్సు", "Tamil": "வயது", "Marathi": "वय", "Bengali": "বয়স"},
    "sex": {"English": "Sex", "Hindi": "लिंग", "Kannada": "ಲಿಂಗ", "Telugu": "లింగం", "Tamil": "பாலினம்", "Marathi": "लिंग", "Bengali": "লিঙ্গ"},
    "cp": {"English": "Chest Pain Type (0-3)", "Hindi": "छाती में दर्द प्रकार (0-3)", "Kannada": "ಎದೆ ನೋವಿನ ಪ್ರಕಾರ", "Telugu": "ఛాతీ నొప్పి రకం", "Tamil": "நெஞ்சு வலி வகை", "Marathi": "छातीतील दुखण्याचा प्रकार", "Bengali": "বুকের ব্যথার ধরণ"},
    "trestbps": {"English": "Resting Blood Pressure (mmHg)", "Hindi": "रक्तचाप (mmHg)", "Kannada": "ರಕ್ತದೊತ್ತಡ", "Telugu": "రక్తపోటు", "Tamil": "இரத்த அழுத்தம்", "Marathi": "रक्तदाब", "Bengali": "রক্তচাপ"},
    "chol": {"English": "Serum Cholesterol (mg/dl)", "Hindi": "कोलेस्ट्रॉल (mg/dl)", "Kannada": "ಕೊಲೆಸ್ಟ್ರಾಲ್", "Telugu": "కొలెస్ట్రాల్", "Tamil": "கொலஸ்ட்ரால்", "Marathi": "कोलेस्ट्रॉल", "Bengali": "কোলেস্টেরল"},
    "fbs": {"English": "Fasting Blood Sugar > 120 mg/dl", "Hindi": "फास्टिंग शुगर > 120", "Kannada": "ಉಪವಾಸದ ಸಕ್ಕರೆ", "Telugu": "ఫాస్టింగ్ బ్లడ్ షుగర్", "Tamil": "உபவாச சர்க்கரை", "Marathi": "उपाशीपोटी साखर", "Bengali": "উপবাস শর্করা"},
    "restecg": {"English": "Resting ECG Results (0-2)", "Hindi": "ईसीजी परिणाम (0-2)", "Kannada": "ಇಸಿಜಿ ಫಲಿತಾಂಶ", "Telugu": "ఈసీజీ ఫలితం", "Tamil": "ஈசிஜி முடிவு", "Marathi": "ईसीजी निकाल", "Bengali": "ইসিজি ফলাফল"},
    "thalach": {"English": "Max Heart Rate Achieved (bpm)", "Hindi": "अधिकतम हृदय गति (bpm)", "Kannada": "ಗರಿಷ್ಠ ಹೃದಯ ಬಡಿತ", "Telugu": "గరిష్ట గుండె వేగం", "Tamil": "அதிகபட்ச இதய துடிப்பு", "Marathi": "कमाल हृदयगती", "Bengali": "সর্বোচ্চ হৃদস্পন্দন"},
    "exang": {"English": "Exercise Induced Angina", "Hindi": "व्यायाम प्रेरित दर्द (Angina)", "Kannada": "ವ್ಯಾಯಾಮದ ಎದೆನೋವು", "Telugu": "వ్యాయామ ఆంజినా", "Tamil": "உடற்பயிற்சி நெஞ்சு வலி", "Marathi": "व्यायामाने छातीतील दुखणे", "Bengali": "ব্যায়াম জনিত বুকে ব্যথা"},
    "oldpeak": {"English": "ST Depression (mm)", "Hindi": "एसटी डिप्रेशन (mm)", "Kannada": "ಎಸ್ಟಿ ಖಿನ್ನತೆ", "Telugu": "ST డిప్రెషన్", "Tamil": "ST டிப்ரஷன்", "Marathi": "एसटी डिप्रेशन", "Bengali": "এসটি ডিপ্রেশন"},
    "slope": {"English": "ST Slope (0-2)", "Hindi": "एसटी ढलान (0-2)", "Kannada": "ST ಇಳಿಜಾರು", "Telugu": "ST స్లోప్", "Tamil": "ST சாய்வு", "Marathi": "एसटी उतर", "Bengali": "এসটি ঢাল"},
    "ca": {"English": "Major Vessels Colored (0-3)", "Hindi": "मुख्य धमनियां (0-3)", "Kannada": "ಪ್ರಮುಖ ರಕ್ತನಾಳಗಳು", "Telugu": "ప్రధాన రక్తనాళాలు", "Tamil": "முக்கிய இரத்த நாளங்கள்", "Marathi": "मुख्य रक्तवाहिन्या", "Bengali": "প্রধান রক্তনালীসমূহ"}
}


# ---------------------------------------------------------
# 2. SESSION STATE & INITIAL DATABASE
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
            "shap_explanation": None
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
            "shap_explanation": None
        },
        "P103": {
            "name": "Suresh Kumar",
            "pass": "pass123",
            "age": 64,
            "sex": 1,
            "cp": 3,
            "trestbps": 160,
            "chol": 285,
            "fbs": 1,
            "restecg": 2,
            "thalach": 115,
            "exang": 1,
            "oldpeak": 3.1,
            "slope": 0,
            "ca": 3,
            "risk": None,
            "shap_explanation": None
        }
    }

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "active_patient_id" not in st.session_state:
    st.session_state.active_patient_id = "P101"
if "logged_patient_id" not in st.session_state:
    st.session_state.logged_patient_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------
# 3. SIDEBAR LANGUAGE & CONFIGURATION
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=70)
st.sidebar.title("🌐 Language / भाषा")
lang = st.sidebar.selectbox(
    "Select Language:",
    ["English", "Hindi", "Kannada", "Telugu", "Tamil", "Marathi", "Bengali"]
)
t = TRANSLATIONS[lang]

st.sidebar.markdown("---")
user_key = st.sidebar.text_input(t["api_key_prompt"], type="password", key="custom_api_key")

# Helper function to safely fetch Streamlit secrets without raising missing file errors
def get_streamlit_secret(key):
    try:
        return st.secrets.get(key)
    except Exception:
        return None

API_KEY = user_key or get_streamlit_secret("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")


# ---------------------------------------------------------
# 4. LOAD ML MODELS (Risk Detector + Cause Explainer)
# ---------------------------------------------------------
@st.cache_resource
def load_ml_models():
    risk_model = None
    cause_explainer = None
    
    # Model 1: Risk Detection Model
    for f in ["cardio_model.pkl", "cardio.plk"]:
        if os.path.exists(f):
            try:
                risk_model = joblib.load(f)
                break
            except Exception as e:
                pass
                
    # Model 2: Cause Explainer (SHAP) Model
    for f in ["shap_explainer.pkl", "sharp_explainer.plk"]:
        if os.path.exists(f):
            try:
                cause_explainer = joblib.load(f)
                break
            except Exception as e:
                pass
                
    return risk_model, cause_explainer

risk_model, cause_explainer = load_ml_models()


# Helper function to convert input values to DataFrame
def prepare_patient_df(p_dict):
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca"
    ]
    vals = [p_dict[col] for col in feature_names]
    df = pd.DataFrame([vals], columns=feature_names)
    return df


# ---------------------------------------------------------
# 5. AUTHENTICATION COMPONENT
# ---------------------------------------------------------
if not st.session_state.authenticated:
    st.markdown(f"<div class='main-header'>{t['login_header']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>{t['subtitle']}</div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs([f"👤 {t['patient_login']}", f"👨‍⚕️ {t['doctor_login']}"])
    
    with tab1:
        st.subheader(t["patient_login"])
        p_id = st.text_input(t["patient_id"], value="P101", key="login_p_id")
        p_pass = st.text_input(t["password"], type="password", value="pass123", key="login_p_pass")
        if st.button(t["login_patient_btn"], type="primary"):
            if p_id in st.session_state.database and st.session_state.database[p_id]["pass"] == p_pass:
                st.session_state.authenticated = True
                st.session_state.user_role = "Patient"
                st.session_state.logged_patient_id = p_id
                st.session_state.active_patient_id = p_id
                st.session_state.chat_history = []
                st.rerun()
            else:
                st.error(t["invalid_creds"])
                
    with tab2:
        st.subheader(t["doctor_login"])
        d_id = st.text_input(t["doctor_id"], value="DOC01", key="login_d_id")
        d_pass = st.text_input(t["password"], type="password", value="doc123", key="login_d_pass")
        if st.button(t["login_doctor_btn"], type="primary"):
            if d_id == "DOC01" and d_pass == "doc123":
                st.session_state.authenticated = True
                st.session_state.user_role = "Doctor"
                st.session_state.active_patient_id = list(st.session_state.database.keys())[0]
                st.session_state.chat_history = []
                st.rerun()
            else:
                st.error(t["invalid_creds"])
                
    st.stop()


# Sidebar Logout & Active Session Info
st.sidebar.markdown(f"**Logged in as:** {t['role_' + st.session_state.user_role.lower()]}")
if st.session_state.user_role == "Patient":
    p_name = st.session_state.database[st.session_state.logged_patient_id]["name"]
    st.sidebar.info(f"👤 Patient: **{p_name}** ({st.session_state.logged_patient_id})")

if st.sidebar.button(t["logout"]):
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.logged_patient_id = None
    st.session_state.chat_history = []
    st.rerun()

st.markdown(f"<div class='main-header'>{t['title']}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 6. DASHBOARDS (PATIENT VS DOCTOR)
# ---------------------------------------------------------

# FUNCTION: Render Vitals Form
def render_vitals_form(p_id, allow_edit=True):
    p_data = st.session_state.database[p_id]
    st.subheader(f"📍 Clinical Vitals Profile — {p_data['name']} (ID: {p_id})")
    
    with st.form(key=f"vitals_form_{p_id}"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input(FEATURE_LABELS["age"][lang], 18, 110, int(p_data["age"]), disabled=not allow_edit)
            sex = st.selectbox(FEATURE_LABELS["sex"][lang], [0, 1], index=int(p_data["sex"]), format_func=lambda x: "Female" if x==0 else "Male", disabled=not allow_edit)
            cp = st.selectbox(FEATURE_LABELS["cp"][lang], [0, 1, 2, 3], index=int(p_data["cp"]), disabled=not allow_edit, help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
            trestbps = st.number_input(FEATURE_LABELS["trestbps"][lang], 80, 240, int(p_data["trestbps"]), disabled=not allow_edit)

        with c2:
            chol = st.number_input(FEATURE_LABELS["chol"][lang], 100, 600, int(p_data["chol"]), disabled=not allow_edit)
            fbs = st.selectbox(FEATURE_LABELS["fbs"][lang], [0, 1], index=int(p_data["fbs"]), format_func=lambda x: "No (<=120)" if x==0 else "Yes (>120)", disabled=not allow_edit)
            restecg = st.selectbox(FEATURE_LABELS["restecg"][lang], [0, 1, 2], index=int(p_data["restecg"]), disabled=not allow_edit, help="0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy")
            thalach = st.number_input(FEATURE_LABELS["thalach"][lang], 50, 230, int(p_data["thalach"]), disabled=not allow_edit)

        with c3:
            exang = st.selectbox(FEATURE_LABELS["exang"][lang], [0, 1], index=int(p_data["exang"]), format_func=lambda x: "No" if x==0 else "Yes", disabled=not allow_edit)
            oldpeak = st.number_input(FEATURE_LABELS["oldpeak"][lang], 0.0, 7.0, float(p_data["oldpeak"]), step=0.1, disabled=not allow_edit)
            slope = st.selectbox(FEATURE_LABELS["slope"][lang], [0, 1, 2], index=int(p_data["slope"]), disabled=not allow_edit, help="0: Upsloping, 1: Flat, 2: Downsloping")
            ca = st.selectbox(FEATURE_LABELS["ca"][lang], [0, 1, 2, 3], index=int(p_data["ca"]), disabled=not allow_edit)

        submit_btn = st.form_submit_button(t["update_vitals"], type="primary") if allow_edit else False

        if submit_btn:
            p_data["age"] = age
            p_data["sex"] = sex
            p_data["cp"] = cp
            p_data["trestbps"] = trestbps
            p_data["chol"] = chol
            p_data["fbs"] = fbs
            p_data["restecg"] = restecg
            p_data["thalach"] = thalach
            p_data["exang"] = exang
            p_data["oldpeak"] = oldpeak
            p_data["slope"] = slope
            p_data["ca"] = ca
            p_data["risk"] = None # reset calculated risk to trigger re-screening
            p_data["shap_explanation"] = None
            st.success(t["vitals_saved"])
            st.rerun()


# FUNCTION: Render Model #1 & Model #2 Execution Panel
def render_ml_analysis_panel(p_id):
    p_data = st.session_state.database[p_id]
    input_df = prepare_patient_df(p_data)
    
    col_a, col_b = st.columns(2)
    
    # MODEL #1: RISK DETECTION
    with col_a:
        st.markdown(f"### {t['model1_header']}")
        if st.button(t["run_screening"], key=f"run_model1_{p_id}", type="primary"):
            if risk_model is not None:
                try:
                    # align features if model requires specific boosters
                    booster_features = getattr(risk_model, "feature_names_in_", None)
                    if booster_features is not None:
                        input_df_aligned = input_df[booster_features]
                    else:
                        input_df_aligned = input_df
                        
                    prob = risk_model.predict_proba(input_df_aligned)[0][1] * 100
                    p_data["risk"] = prob
                except Exception as err:
                    st.error(f"Error running Model #1: {err}")
            else:
                st.error("Model #1 (cardio_model.pkl) not loaded.")

        if p_data["risk"] is not None:
            risk_val = p_data["risk"]
            st.metric(t["risk_score"], f"{risk_val:.1f}%")
            
            if risk_val >= 65:
                st.markdown(f"<div class='high-risk-badge'>{t['high_risk']} ({risk_val:.1f}%)</div>", unsafe_allow_html=True)
            elif risk_val >= 35:
                st.markdown(f"<div class='mod-risk-badge'>{t['mod_risk']} ({risk_val:.1f}%)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='low-risk-badge'>{t['low_risk']} ({risk_val:.1f}%)</div>", unsafe_allow_html=True)
            st.progress(min(1.0, risk_val / 100.0))
        else:
            st.info("Click button above to calculate ML Heart Attack Risk.")

    # MODEL #2: CAUSE EXPLAINER (SHAP)
    with col_b:
        st.markdown(f"### {t['model2_header']}")
        if st.button(t["run_explanation"], key=f"run_model2_{p_id}"):
            if cause_explainer is not None:
                try:
                    shap_vals = cause_explainer.shap_values(input_df)
                    if isinstance(shap_vals, list):
                        shap_array = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
                    elif len(shap_vals.shape) == 2:
                        shap_array = shap_vals[0]
                    else:
                        shap_array = shap_vals
                    p_data["shap_explanation"] = dict(zip(input_df.columns, shap_array))
                except Exception as err:
                    st.error(f"Error running Model #2 SHAP explainer: {err}")
            else:
                st.error("Model #2 (shap_explainer.pkl) not loaded.")

        if p_data["shap_explanation"] is not None:
            shap_dict = p_data["shap_explanation"]
            
            # Sort factors by impact
            sorted_factors = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
            risk_increasing = [f for f in sorted_factors if f[1] > 0]
            risk_decreasing = [f for f in sorted_factors if f[1] < 0]
            
            st.markdown(f"**{t['top_causes']}**")
            for feat, val in risk_increasing[:3]:
                fname = FEATURE_LABELS.get(feat, {}).get(lang, feat)
                pval = p_data[feat]
                st.write(f"• **{fname}** (Value: `{pval}`): +{val:.2f} risk attribution score")
                
            if risk_decreasing:
                st.markdown(f"**{t['protective']}**")
                for feat, val in sorted(risk_decreasing, key=lambda x: x[1])[:2]:
                    fname = FEATURE_LABELS.get(feat, {}).get(lang, feat)
                    pval = p_data[feat]
                    st.write(f"• **{fname}** (Value: `{pval}`): {val:.2f} protective score")
        else:
            st.info("Click button above to analyze causes driving heart attack risk.")

    # RENDER SHAP CAUSE BREAKDOWN CHART IF AVAILABLE
    if p_data["shap_explanation"] is not None:
        st.markdown("---")
        st.subheader(t["shap_chart"])
        
        shap_dict = p_data["shap_explanation"]
        feats = [FEATURE_LABELS.get(k, {}).get(lang, k) for k in shap_dict.keys()]
        scores = list(shap_dict.values())
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        colors = ['#EF4444' if s > 0 else '#10B981' for s in scores]
        bars = ax.barh(feats, scores, color=colors)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel("SHAP Impact Score on Heart Attack Risk (Positive = Increases Risk, Negative = Lowers Risk)")
        ax.set_title(f"Model #2 Cause Attribution Breakdown for {p_data['name']}")
        plt.tight_layout()
        st.pyplot(fig)


# ---------------------------------------------------------
# PATIENT DASHBOARD (STRICT PRIVACY: ONLY LOGGED PATIENT)
# ---------------------------------------------------------
if st.session_state.user_role == "Patient":
    p_id = st.session_state.logged_patient_id
    st.header(t["patient_dash"])
    
    ptab1, ptab2 = st.tabs([t["my_records"], t["chat_header"]])
    
    with ptab1:
        render_vitals_form(p_id, allow_edit=True)
        st.markdown("---")
        render_ml_analysis_panel(p_id)
        
    with ptab2:
        pass # Chat rendered below for active session


# ---------------------------------------------------------
# DOCTOR DASHBOARD (ALL PATIENTS + SELECT + ADD NEW)
# ---------------------------------------------------------
elif st.session_state.user_role == "Doctor":
    st.header(t["doctor_dash"])
    
    dtab1, dtab2, dtab3 = st.tabs([t["all_patients"], "🔬 Patient Clinical Review & Diagnosis", t["add_patient"]])
    
    # TAB 1: ALL PATIENTS DIRECTORY OVERVIEW
    with dtab1:
        st.subheader(t["all_patients"])
        
        records = []
        for pid, data in st.session_state.database.items():
            r_str = f"{data['risk']:.1f}%" if data["risk"] is not None else "Not Screened"
            records.append({
                "Patient ID": pid,
                "Name": data["name"],
                "Age": data["age"],
                "Gender": "Male" if data["sex"]==1 else "Female",
                "Resting BP (mmHg)": data["trestbps"],
                "Cholesterol (mg/dl)": data["chol"],
                "ST Depression": data["oldpeak"],
                "Calculated Risk Score": r_str
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True)
        
    # TAB 2: SELECT & REVIEW SPECIFIC PATIENT
    with dtab2:
        selected_p = st.selectbox(
            t["select_patient"],
            options=list(st.session_state.database.keys()),
            format_func=lambda pid: f"{pid} - {st.session_state.database[pid]['name']}",
            index=list(st.session_state.database.keys()).index(st.session_state.active_patient_id)
        )
        st.session_state.active_patient_id = selected_p
        p_id = st.session_state.active_patient_id
        
        render_vitals_form(p_id, allow_edit=True)
        st.markdown("---")
        render_ml_analysis_panel(p_id)

    # TAB 3: REGISTER NEW PATIENT
    with dtab3:
        st.subheader(t["add_patient"])
        with st.form("new_patient_form"):
            col1, col2 = st.columns(2)
            with col1:
                n_id = st.text_input(t["new_p_id"], value=f"P10{len(st.session_state.database)+1}")
                n_name = st.text_input(t["new_p_name"], value="New Patient")
                n_pass = st.text_input(t["new_p_pass"], type="password", value="pass123")
                n_age = st.number_input(FEATURE_LABELS["age"]["English"], 18, 100, 50)
                n_sex = st.selectbox(FEATURE_LABELS["sex"]["English"], [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
                n_cp = st.selectbox(FEATURE_LABELS["cp"]["English"], [0, 1, 2, 3], index=1)
                n_trestbps = st.number_input(FEATURE_LABELS["trestbps"]["English"], 80, 220, 130)

            with col2:
                n_chol = st.number_input(FEATURE_LABELS["chol"]["English"], 100, 500, 220)
                n_fbs = st.selectbox(FEATURE_LABELS["fbs"]["English"], [0, 1], index=0)
                n_restecg = st.selectbox(FEATURE_LABELS["restecg"]["English"], [0, 1, 2], index=0)
                n_thalach = st.number_input(FEATURE_LABELS["thalach"]["English"], 60, 220, 150)
                n_exang = st.selectbox(FEATURE_LABELS["exang"]["English"], [0, 1], index=0)
                n_oldpeak = st.number_input(FEATURE_LABELS["oldpeak"]["English"], 0.0, 6.0, 1.0)
                n_slope = st.selectbox(FEATURE_LABELS["slope"]["English"], [0, 1, 2], index=1)
                n_ca = st.selectbox(FEATURE_LABELS["ca"]["English"], [0, 1, 2, 3], index=0)

            add_btn = st.form_submit_button(t["create_p_btn"], type="primary")
            if add_btn:
                if n_id in st.session_state.database:
                    st.error("Patient ID already exists!")
                else:
                    st.session_state.database[n_id] = {
                        "name": n_name,
                        "pass": n_pass,
                        "age": n_age,
                        "sex": n_sex,
                        "cp": n_cp,
                        "trestbps": n_trestbps,
                        "chol": n_chol,
                        "fbs": n_fbs,
                        "restecg": n_restecg,
                        "thalach": n_thalach,
                        "exang": n_exang,
                        "oldpeak": n_oldpeak,
                        "slope": n_slope,
                        "ca": n_ca,
                        "risk": None,
                        "shap_explanation": None
                    }
                    st.session_state.active_patient_id = n_id
                    st.success(f"✅ Patient {n_name} ({n_id}) registered successfully!")
                    st.rerun()


# ---------------------------------------------------------
# 7. INTERACTIVE LLM CLINICAL ASSISTANT (MULTI-LINGUAL)
# ---------------------------------------------------------
st.markdown("---")
st.header(t["chat_header"])

current_pid = (
    st.session_state.logged_patient_id 
    if st.session_state.user_role == "Patient" 
    else st.session_state.active_patient_id
)
curr_pdata = st.session_state.database[current_pid]

# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(t["chat_placeholder"]):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare detailed medical context for LLM
    risk_val_str = (
        f"{curr_pdata['risk']:.1f}%" if curr_pdata['risk'] is not None else "Not Screened Yet"
    )
    
    top_causes_str = "Not Analyzed Yet"
    if curr_pdata.get("shap_explanation"):
        sorted_f = sorted(curr_pdata["shap_explanation"].items(), key=lambda x: x[1], reverse=True)
        top_causes_str = ", ".join([f"{k} (impact: +{v:.2f})" for k, v in sorted_f if v > 0][:3])

    system_prompt = f"""
    You are an expert AI Cardiac Clinical Consultant inside the CardioGuard medical platform.
    
    CURRENT ACTIVE PATIENT CONTEXT:
    - Patient Name: {curr_pdata['name']} (ID: {current_pid})
    - Age: {curr_pdata['age']}, Gender: {'Male' if curr_pdata['sex']==1 else 'Female'}
    - Resting Blood Pressure: {curr_pdata['trestbps']} mmHg
    - Serum Cholesterol: {curr_pdata['chol']} mg/dl
    - Fasting Blood Sugar > 120: {'Yes' if curr_pdata['fbs']==1 else 'No'}
    - Max Heart Rate Achieved: {curr_pdata['thalach']} bpm
    - Exercise Induced Angina: {'Yes' if curr_pdata['exang']==1 else 'No'}
    - ST Depression (oldpeak): {curr_pdata['oldpeak']}
    - Model #1 Calculated Heart Attack Risk: {risk_val_str}
    - Model #2 Top SHAP Cause Drivers: {top_causes_str}
    
    INSTRUCTIONS:
    1. Respond STRICTLY in the target language: {lang}.
    2. Provide clear, empathetic, medically sound, and structured responses with Markdown formatting.
    3. Refer to the patient's specific health metrics and Model #1 risk score / Model #2 causes when relevant.
    """

    full_llm_query = f"{system_prompt}\n\nUSER PROMPT: {prompt}"

    with st.chat_message("assistant"):
        with st.spinner("Analyzing patient metrics and generating response..."):
            reply = None
            
            # Try new Google GenAI SDK if API_KEY is present
            if API_KEY:
                try:
                    if GENAI_NEW_AVAILABLE:
                        client = genai.Client(api_key=API_KEY)
                        res = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=full_llm_query
                        )
                        reply = res.text
                    elif GENAI_LEGACY_AVAILABLE:
                        genai_legacy.configure(api_key=API_KEY)
                        model = genai_legacy.GenerativeModel("gemini-1.5-flash")
                        res = model.generate_content(full_llm_query)
                        reply = res.text
                except Exception as err:
                    reply = f"⚠️ Gemini API Error ({type(err).__name__}): {str(err)}"

            # Robust Offline Intelligent Clinical Response Generator if API Key is not set or fails
            if not reply:
                reply = (
                    f"**[CardioGuard AI Clinical Assistant — Response for {curr_pdata['name']}]**\n\n"
                    f"**Patient Health Summary ({lang}):**\n"
                    f"- **Age / Gender:** {curr_pdata['age']} | {'Male' if curr_pdata['sex']==1 else 'Female'}\n"
                    f"- **Blood Pressure:** {curr_pdata['trestbps']} mmHg\n"
                    f"- **Cholesterol:** {curr_pdata['chol']} mg/dl\n"
                    f"- **Heart Attack Risk Level:** {risk_val_str}\n\n"
                    f"**Clinical Recommendation:**\n"
                    f"Based on your query regarding *'{prompt}'*, it is recommended to monitor resting blood pressure "
                    f"and keep cholesterol levels within optimal limits (<200 mg/dl). Maintain a low-sodium heart-healthy diet "
                    f"and perform regular moderate cardio exercises as recommended by your physician.\n\n"
                    f"*(Tip: To enable live dynamic LLM responses, enter your GEMINI_API_KEY in the sidebar)*"
                )