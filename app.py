import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Import the new LangGraph agent workflow
from agentic_workflow import chat_with_agent

# --- SET UP DIRECTORIES ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- PAGE CONFIGURATION & CUSTOM CSS (WOW FACTOR) ---
st.set_page_config(
    page_title="MediRisk | Healthcare Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium dark-glassmorphism aesthetic
premium_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Professional Glassmorphic Cards */
.glass-card {
    border-radius: 16px;
    padding: 28px;
    background: var(--secondary-background-color);
    border: 1px solid rgba(150, 150, 150, 0.1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
}

/* Beautiful Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
}
.stTabs [data-baseweb="tab"] {
    height: 54px;
    border-radius: 12px 12px 0px 0px;
    padding: 12px 24px;
    font-weight: 500;
    background-color: var(--secondary-background-color);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(150, 150, 150, 0.1);
    border-bottom: none;
    opacity: 0.7;
}

.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    border-bottom: 3px solid #4361ee !important;
    color: #4361ee !important;
    opacity: 1;
}

div[data-testid="stMetricValue"] {
    color: var(--primary-color) !important;
    font-weight: 600 !important;
}

/* Primary Button Styling with Micro-animation */
button[kind="primary"] {
    background: linear-gradient(135deg, #4361ee, #3f37c9);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 15px rgba(67, 97, 238, 0.2);
}

button[kind="primary"]:hover {
    box-shadow: 0 8px 25px rgba(67, 97, 238, 0.4);
    transform: translateY(-2px);
}

.stHeader {
    background: transparent !important;
}

h1.main-title {
    text-align: center;
    background: linear-gradient(45deg, #4cc9f0, #4361ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 10px;
    font-weight: 800;
    font-size: 3.5rem;
    letter-spacing: -1.5px;
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
    y_clf_prob = log_reg.predict_proba(X_test_scaled_clf)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test_clf, y_clf_prob)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        'accuracy': accuracy_score(y_test_clf, y_clf_pred),
        'precision': precision_score(y_test_clf, y_clf_pred),
        'recall': recall_score(y_test_clf, y_clf_pred),
        'f1': f1_score(y_test_clf, y_clf_pred),
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'conf_matrix': confusion_matrix(y_test_clf, y_clf_pred),
        'class_report': classification_report(y_test_clf, y_clf_pred, output_dict=True),
        'coefficients': log_reg.coef_[0]
    }
    return log_reg, dt_clf, scaler_clf, le_gender, features_clf, metrics

# --- LOAD DATA ---
FILE_PATH = 'synthetic_clinical_dataset.csv'
raw_df = load_data(FILE_PATH)

st.markdown("<h1 class='main-title'>MediRisk Agentic Health System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: var(--text-color); opacity: 0.7; font-size: 1.1rem; margin-top: -10px;'>Predictive Risk Modeling integrated with RAG-powered Agentic Support</p>", unsafe_allow_html=True)
st.write("---")

if raw_df is None:
    st.error(f"Dataset file '{FILE_PATH}' not found in root directory.")
    st.stop()

df = preprocess_data(raw_df)
log_reg, dt_clf, scaler_clf, le_gender, features_clf, metrics = train_models(df)


# --- TABS LAYOUT ---
tab_dash, tab1, tab2, tab3 = st.tabs(["Dashboard", "Phase 1: AI Predictor", "Phase 2: Agentic Chat", "Model Telemetry"])

# ================================
# TAB 0: DASHBOARD
# ================================
with tab_dash:
    st.subheader("Global Project Metrics", anchor=False)
    
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    metric_style = "text-align:center; padding: 25px 10px; border-radius: 16px; background: var(--secondary-background-color); border: 1px solid rgba(150,150,150,0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.05);"
    with dcol1:
        st.markdown(f"<div style='{metric_style}'><h1 style='margin:0; color:#4361ee; font-size:3.5rem;'>{metrics['accuracy']*100:.1f}%</h1><p style='margin:0; font-weight:600; opacity:0.7; letter-spacing:1px;'>ACCURACY</p></div>", unsafe_allow_html=True)
    with dcol2:
        st.markdown(f"<div style='{metric_style}'><h1 style='margin:0; color:#4361ee; font-size:3.5rem;'>{metrics['precision']*100:.1f}%</h1><p style='margin:0; font-weight:600; opacity:0.7; letter-spacing:1px;'>PRECISION</p></div>", unsafe_allow_html=True)
    with dcol3:
        st.markdown(f"<div style='{metric_style}'><h1 style='margin:0; color:#4361ee; font-size:3.5rem;'>{metrics['recall']*100:.1f}%</h1><p style='margin:0; font-weight:600; opacity:0.7; letter-spacing:1px;'>RECALL</p></div>", unsafe_allow_html=True)
    with dcol4:
        st.markdown(f"<div style='{metric_style}'><h1 style='margin:0; color:#4361ee; font-size:3.5rem;'>{metrics['f1']*100:.1f}%</h1><p style='margin:0; font-weight:600; opacity:0.7; letter-spacing:1px;'>F1 SCORE</p></div>", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Pipeline Architecture", anchor=False)
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    step_style = "text-align:center; padding: 25px 15px; min-height: 180px; border-radius: 16px; background: var(--secondary-background-color); border: 1px solid rgba(67, 97, 238, 0.3); box-shadow: 0 4px 15px rgba(67, 97, 238, 0.05);"
    
    with pcol1:
        st.markdown(f"<div style='{step_style}'><div style='background:linear-gradient(135deg, #4361ee, #3f37c9); color:white; width:45px; height:45px; border-radius:50%; line-height:45px; margin:0 auto 15px auto; font-weight:bold; box-shadow: 0 4px 10px rgba(67,97,238,0.4);'>1</div><h4 style='margin:0; font-weight:700;'>Input</h4><p style='font-size:0.85rem; opacity:0.8; margin-top:8px;'>Configure patient vitals via clinical predictor</p></div>", unsafe_allow_html=True)
    with pcol2:
        st.markdown(f"<div style='{step_style}'><div style='background:linear-gradient(135deg, #4361ee, #3f37c9); color:white; width:45px; height:45px; border-radius:50%; line-height:45px; margin:0 auto 15px auto; font-weight:bold; box-shadow: 0 4px 10px rgba(67,97,238,0.4);'>2</div><h4 style='margin:0; font-weight:700;'>ML Pipeline</h4><p style='font-size:0.85rem; opacity:0.8; margin-top:8px;'>Feature engineering, scaling & classification inference</p></div>", unsafe_allow_html=True)
    with pcol3:
        st.markdown(f"<div style='{step_style}'><div style='background:linear-gradient(135deg, #4361ee, #3f37c9); color:white; width:45px; height:45px; border-radius:50%; line-height:45px; margin:0 auto 15px auto; font-weight:bold; box-shadow: 0 4px 10px rgba(67,97,238,0.4);'>3</div><h4 style='margin:0; font-weight:700;'>Risk Analysis</h4><p style='font-size:0.85rem; opacity:0.8; margin-top:8px;'>Mortality probability logic & categorical extraction</p></div>", unsafe_allow_html=True)
    with pcol4:
        st.markdown(f"<div style='{step_style}'><div style='background:linear-gradient(135deg, #4361ee, #3f37c9); color:white; width:45px; height:45px; border-radius:50%; line-height:45px; margin:0 auto 15px auto; font-weight:bold; box-shadow: 0 4px 10px rgba(67,97,238,0.4);'>4</div><h4 style='margin:0; font-weight:700;'>AI Agent</h4><p style='font-size:0.85rem; opacity:0.8; margin-top:8px;'>LangGraph + FAISS generates tailored health strategy</p></div>", unsafe_allow_html=True)


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
        
        submitted = st.form_submit_button("Engage Predictive Models")

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
            st.markdown(f"<div class='glass-card' style='border-left: 6px solid {p_color};'>"
                        f"<p style='margin:0; font-size:0.9rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; opacity:0.8;'>Mortality Risk Score</p>"
                        f"<h1 style='margin:0; color:{p_color}; font-size:3.5rem; font-weight:800;'>{risk_score:.1f}%</h1>"
                        f"</div>", unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"<div class='glass-card' style='border-left: 6px solid {p_color};'>"
                        f"<p style='margin:0; font-size:0.9rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; opacity:0.8;'>Assessment Level</p>"
                        f"<h1 style='margin:0; color:{p_color}; font-size:3.5rem; font-weight:800;'>{predicted_level}</h1>"
                        f"</div>", unsafe_allow_html=True)
            
        st.success("ML Analysis Complete! Head over to Phase 2: Agentic Chat to receive an autonomous RAG-powered healthcare plan.")

# ================================
# TAB 2: PHASE 2 AGENTIC CHAT
# ================================
with tab2:
    st.subheader("LangGraph Agentic Support", anchor=False)
    
    if st.session_state.risk_score is None:
        st.info("Please complete the Phase 1 Predictive Assessment first to initialize the Agent context.")
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
                    
                    try:
                        resp_meta = chat_with_agent(context, risk, prompt, st.session_state.chat_history[:-1])
                        st.session_state.chat_history.append(resp_meta)
                    except Exception as e:
                        st.error(f"Agent Service Unavailable: Please check API keys or vector index. Details: {str(e)}")
                        st.session_state.chat_history.pop() # Remove the prompt if it failed
                    st.rerun()

        if user_query := st.chat_input("Ask the AI about treatment, diet, or risks..."):
            st.chat_message("user").write(user_query)
            
            with st.spinner("LangGraph Agent is processing..."):
                context = st.session_state.patient_context
                risk = st.session_state.risk_score
                from langchain_core.messages import HumanMessage
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                
                try:
                    resp_meta = chat_with_agent(context, risk, user_query, st.session_state.chat_history[:-1])
                    st.session_state.chat_history.append(resp_meta)
                except Exception as e:
                    st.error(f"Agent Service Unavailable: Please check API keys or vector index. Details: {str(e)}")
                    st.session_state.chat_history.pop() # Remove the message if generation failed
                st.rerun()

# ================================
# TAB 3: TELEMETRY (Model Stats)
# ================================
with tab3:
    st.subheader("In-Depth Model Telemetry", anchor=False)
    
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("##### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Survived', 'Deceased'],
                    yticklabels=['Survived', 'Deceased'], ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)
        
        st.markdown("##### Full Classification Report")
        cr_df = pd.DataFrame(metrics['class_report']).transpose()
        st.dataframe(cr_df.style.background_gradient(cmap='Blues').format(precision=3), use_container_width=True)
        
    with tcol2:
        st.markdown("##### ROC Curve (AUC = {:.3f})".format(metrics['roc_auc']))
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(metrics['fpr'], metrics['tpr'], color='#4361ee', lw=2, label=f"ROC curve (area = {metrics['roc_auc']:.2f})")
        ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        
        st.markdown("##### Feature Importance (Logistic Reg Coefficients)")
        coefs = metrics['coefficients']
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        colors = ['#4cc9f0' if c > 0 else '#f72585' for c in coefs[sorted_idx]]
        ax_fi.barh([features_clf[i] for i in sorted_idx], np.abs(coefs[sorted_idx]), color=colors)
        ax_fi.set_xlabel('Coefficient Impact')
        ax_fi.invert_yaxis()
        st.pyplot(fig_fi)
