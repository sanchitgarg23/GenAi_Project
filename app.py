import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    confusion_matrix, classification_report, mean_absolute_error
)

# Import the new LangGraph agent workflow
from agentic_workflow import chat_with_agent

# --- SET UP DIRECTORIES ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- PAGE CONFIGURATION & CUSTOM CSS (WOW FACTOR) ---
st.set_page_config(
    page_title="MediRisk | Agentic Healthcare",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium dark-glassmorphism aesthetic
premium_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0b132b, #1c2541, #0a0f25);
    color: #e0e6ed;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #121c3b, #0d142b);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: rgba(25, 35, 65, 0.6);
    border-radius: 8px 8px 0px 0px;
    gap: 10px;
    padding: 10px 20px;
    color: #8c9eff !important;
    font-weight: 600;
    transition: 0.3s ease;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(60, 85, 170, 0.4) !important;
    border-bottom: 2px solid #5c7cfa !important;
    color: #ffffff !important;
    text-shadow: 0px 0px 10px rgba(92, 124, 250, 0.5);
}

div[data-testid="stMetricValue"] {
    color: #4cd137 !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
}

.st-emotion-cache-1wivap2 {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}

button.st-emotion-cache-12w0qtp {
    background: linear-gradient(135deg, #4361ee, #3a0ca3);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

button.st-emotion-cache-12w0qtp:hover {
    box-shadow: 0px 0px 15px rgba(67, 97, 238, 0.5);
    transform: translateY(-2px);
}

.stHeader {
    background: transparent !important;
}

h1, h2, h3 {
    color: #f8f9fa !important;
    font-weight: 700 !important;
}

h1.st-emotion-cache-10trblm {
    text-align: center;
    background: -webkit-linear-gradient(45deg, #4cc9f0, #4361ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 20px;
}
</style>
"""
st.markdown(premium_css, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "patient_context" not in st.session_state:
    st.session_state.patient_context = ""
if "risk_score" not in st.session_state:
    st.session_state.risk_score = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- CACHED ML FUNCTIONS (PHASE 1) ---
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

@st.cache_data
def preprocess_data(df):
    df = df.rename(columns={
        'age': 'Age', 'sex': 'Gender', 'bmi': 'BMI',
        'systolic_bp': 'Systolic_BP', 'diastolic_bp': 'Diastolic_BP',
        'cholesterol': 'Cholesterol_Total', 'glucose': 'Glucose',
        'creatinine': 'Creatinine', 'diabetes': 'Diabetes', 'hypertension': 'Hypertension'
    })
    cols_to_drop = ['patient_id', 'diagnosis', 'readmission_30d']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.dropna()
    return df

@st.cache_resource
def train_models(df):
    le_gender = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

    y_clf = df['mortality']
    features_clf = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol_Total', 'Glucose', 'Creatinine', 'Diabetes', 'Hypertension', 'Gender_Encoded']
    X_clf = df[features_clf]

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    scaler_clf = StandardScaler()
    X_train_scaled_clf = scaler_clf.fit_transform(X_train_clf)
    X_test_scaled_clf = scaler_clf.transform(X_test_clf)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled_clf, y_train_clf)
    
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_clf.fit(X_train_scaled_clf, y_train_clf)

    # Metrics
    y_clf_pred = log_reg.predict(X_test_scaled_clf)
    metrics = {
        'accuracy': accuracy_score(y_test_clf, y_clf_pred),
        'conf_matrix': confusion_matrix(y_test_clf, y_clf_pred),
        'coefficients': log_reg.coef_[0]
    }
    return log_reg, dt_clf, scaler_clf, le_gender, features_clf, metrics

# --- LOAD DATA ---
FILE_PATH = 'synthetic_clinical_dataset.csv'
raw_df = load_data(FILE_PATH)

st.title("🧬 MediRisk Agentic Health System")
st.markdown("<p style='text-align: center; color: #a1a1aa;'>Predictive Risk Modeling integrated with RAG-powered Agentic Support</p>", unsafe_allow_html=True)
st.write("---")

if raw_df is None:
    st.error(f"Dataset file '{FILE_PATH}' not found in root directory.")
    st.stop()

df = preprocess_data(raw_df)
log_reg, dt_clf, scaler_clf, le_gender, features_clf, metrics = train_models(df)


# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🔮 Phase 1: AI Predictor", "🩺 Phase 2: Agentic Chat", "📊 Model Telemetry"])

# ================================
# TAB 1: PHASE 1 MODELING
# ================================
with tab1:
    st.subheader("Patient Vitals Assessment", anchor=False)
    
    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 100, 50)
            systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
            diabetes_input = st.selectbox("Diabetes", ["No", "Yes"])
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80)
            glucose = st.number_input("Glucose", 50, 300, 100)
        with col3:
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
            cholesterol = st.number_input("Cholesterol", 100, 400, 200)
            creatinine = st.number_input("Creatinine", 0.1, 5.0, 1.0, step=0.1)
            hypertension_input = st.selectbox("Hypertension", ["No", "Yes"])

        diabetes = 1 if diabetes_input == "Yes" else 0
        hypertension = 1 if hypertension_input == "Yes" else 0
        
        submitted = st.form_submit_button("Engage Predictive Models 🚀")

    if submitted:
        gender_encoded = le_gender.transform([gender])[0]
        input_dict = {
            'Age': age, 'BMI': bmi, 'Systolic_BP': systolic_bp, 'Diastolic_BP': diastolic_bp,
            'Cholesterol_Total': cholesterol, 'Glucose': glucose, 'Creatinine': creatinine,
            'Diabetes': diabetes, 'Hypertension': hypertension, 'Gender_Encoded': gender_encoded
        }
        input_data_clf = pd.DataFrame([input_dict])[features_clf]
        input_scaled_clf = scaler_clf.transform(input_data_clf)
        
        predicted_proba = log_reg.predict_proba(input_scaled_clf)[0]
        risk_score = predicted_proba[1] * 100
        
        if risk_score < 40:
            predicted_level = "Low"
            p_color = "#4cd137"
        elif risk_score < 70:
            predicted_level = "Medium"
            p_color = "#fbc531"
        else:
            predicted_level = "High"
            p_color = "#e84118"
            
        st.session_state.risk_score = risk_score
        # Build comprehensive patient context for the Agent string
        st.session_state.patient_context = f"A {age} year old {gender} with a BMI of {bmi}. Vitals: Blood Pressure {systolic_bp}/{diastolic_bp}, Glucose {glucose}, Cholesterol {cholesterol}, Creatinine {creatinine}. Pre-existing conditions: Diabetes={diabetes_input}, Hypertension={hypertension_input}."
        st.session_state.chat_history = [] # Reset chat when new prediction occurs
        
        st.write("---")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"<div style='padding: 20px; border-radius: 12px; background: rgba(0,0,0,0.2); border-left: 5px solid {p_color};'>"
                        f"<h3 style='margin:0;'>Mortality Risk Score</h3>"
                        f"<h1 style='margin:0; color:{p_color}; font-size:48px;'>{risk_score:.1f}%</h1>"
                        f"</div>", unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"<div style='padding: 20px; border-radius: 12px; background: rgba(0,0,0,0.2); border-left: 5px solid {p_color};'>"
                        f"<h3 style='margin:0;'>Assessment Level</h3>"
                        f"<h1 style='margin:0; color:{p_color}; font-size:48px;'>{predicted_level}</h1>"
                        f"</div>", unsafe_allow_html=True)
            
        st.success("✅ ML Analysis Complete! Head over to **Phase 2: Agentic Chat** to receive an autonomous RAG-powered healthcare plan.", icon="🤖")

# ================================
# TAB 2: PHASE 2 AGENTIC CHAT
# ================================
with tab2:
    st.subheader("🤖 LangGraph Agentic Support", anchor=False)
    
    if st.session_state.risk_score is None:
        st.info("⚠️ Please complete the Phase 1 Predictive Assessment first to initialize the Agent context.", icon="ℹ️")
    else:
        st.markdown(f"**Current Patient Profile:** `{st.session_state.patient_context}`")
        st.markdown(f"**Identified Risk Score:** `{st.session_state.risk_score:.1f}%`")
        
        # Display Chat History
        for msg in st.session_state.chat_history:
            role = "user" if msg.type == "human" else "assistant"
            st.chat_message(role).write(msg.content)
            
        # First Time Prompt Button (Optional)
        if len(st.session_state.chat_history) == 0:
            if st.button("Generate Initial Autonomous Health Report"):
                with st.chat_message("user"):
                    st.write("Please generate a detailed health report based on my clinical guidelines.")
                with st.spinner("Agent is reasoning and retrieving guidelines..."):
                    context = st.session_state.patient_context
                    risk = st.session_state.risk_score
                    prompt = "Based on my patient profile and risk score, retrieve guidelines and generate a personalized health management report."
                    from langchain_core.messages import HumanMessage
                    
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    resp_meta = chat_with_agent(context, risk, prompt, st.session_state.chat_history[:-1])
                    st.session_state.chat_history.append(resp_meta)
                    st.rerun()

        if user_query := st.chat_input("Ask the AI about treatment, diet, or risks..."):
            st.chat_message("user").write(user_query)
            
            with st.spinner("LangGraph Agent is processing..."):
                context = st.session_state.patient_context
                risk = st.session_state.risk_score
                from langchain_core.messages import HumanMessage
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                
                resp_meta = chat_with_agent(context, risk, user_query, st.session_state.chat_history[:-1])
                st.session_state.chat_history.append(resp_meta)
                st.rerun()

# ================================
# TAB 3: TELEMETRY (Model Stats)
# ================================
with tab3:
    st.subheader("Model Telemetry & Performance Checks", anchor=False)
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("##### Logistic Regression Metrics")
        st.write(f"**Base Accuracy:** {metrics['accuracy']*100:.2f}%")
        
        st.markdown("##### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
        # Ensure plot has dark background aesthetic
        fig_cm.patch.set_facecolor('#0f172a')
        ax_cm.set_facecolor('#0f172a')
        sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='mako',
                    xticklabels=['Survived', 'Deceased'],
                    yticklabels=['Survived', 'Deceased'], ax=ax_cm)
        ax_cm.tick_params(colors='white')
        ax_cm.set_xlabel('Predicted', color='white')
        ax_cm.set_ylabel('Actual', color='white')
        st.pyplot(fig_cm)
        
    with tcol2:
        st.markdown("##### Feature Importance (Logistic Reg Coefficients)")
        coefs = metrics['coefficients']
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        fig_fi.patch.set_facecolor('#0f172a')
        ax_fi.set_facecolor('#0f172a')
        colors = ['#4cc9f0' if c > 0 else '#f72585' for c in coefs[sorted_idx]]
        ax_fi.barh([features_clf[i] for i in sorted_idx], np.abs(coefs[sorted_idx]), color=colors)
        ax_fi.set_xlabel('Coefficient Impact', color='white')
        ax_fi.tick_params(colors='white')
        ax_fi.invert_yaxis()
        st.pyplot(fig_fi)
