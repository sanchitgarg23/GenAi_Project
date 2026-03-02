import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report, mean_absolute_error
import os

# Set page configuration
st.set_page_config(
    page_title="MediRisk | Patient Assessment",
    page_icon="🏥",
    layout="centered"
)

# --- Custom Medical Dark CSS ---
def inject_custom_css():
    st.markdown('<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">', unsafe_allow_html=True)
    
    st.markdown("""
<style>
/* Dark Theme Colors */
:root {
    --primary-color: #00ced1;
    --bg-color: #0e1117;
    --card-bg: #1f2937;
    --text-color: #ffffff;
    --secondary-text: #a0aec0;
    --success-color: #00ff7f;
    --warning-color: #ffd700;
    --danger-color: #ff4500;
}

.stApp {
    background-color: var(--bg-color);
    color: var(--text-color);
}

/* Compact Header */
.main-header {
    text-align: center;
    color: var(--primary-color);
    padding: 1rem 0;
    border-bottom: 1px solid var(--primary-color);
    margin-bottom: 2rem;
}
.main-header h1 {
    font-size: 2.2rem;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
}
.main-header p {
    font-size: 1rem;
    color: var(--secondary-text);
    margin-top: 5px;
}

/* Inputs Styling Override */
.stNumberInput, .stSelectbox, .stSlider, .stRadio {
    margin-bottom: 0px;
}
label {
    color: var(--text-color) !important;
    font-size: 0.85rem !important;
}

/* Section Headers using Icons */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-color);
}
.material-icons {
    font-size: 1.8rem;
    vertical-align: middle;
}

/* Result Cards */
.result-section {
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.1);
}

.result-container {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1.5rem;
}

.result-card {
    background: linear-gradient(145deg, #1f2937, #111827);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    min-width: 220px;
    border: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.2s;
}
.result-card:hover {
    transform: translateY(-5px);
}

.result-card h4 {
    color: var(--secondary-text);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.9rem;
    margin-bottom: 15px;
}

.big-number {
    font-size: 3.5rem;
    font-weight: 800;
    color: var(--primary-color);
    line-height: 1;
    margin: 0;
}

.risk-label {
    font-size: 2rem;
    font-weight: 800;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* Notification Box */
.notification-box {
    margin-top: 2rem;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(90deg, var(--primary-color), #00e5ff);
    color: #0e1117;
    font-weight: bold;
    width: 100%;
    border-radius: 8px;
    height: 3rem;
    margin-top: 2rem;
    border: none;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 4px 6px rgba(0, 206, 209, 0.2);
}
.stButton>button:hover {
    box-shadow: 0 6px 8px rgba(0, 206, 209, 0.4);
}
</style>
""", unsafe_allow_html=True)

inject_custom_css()

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df

@st.cache_data
def preprocess_data(df):
    # Rename columns to standard format
    df = df.rename(columns={
        'age': 'Age',
        'sex': 'Gender',
        'bmi': 'BMI',
        'systolic_bp': 'Systolic_BP',
        'diastolic_bp': 'Diastolic_BP',
        'cholesterol': 'Cholesterol_Total',
        'glucose': 'Glucose',
        'creatinine': 'Creatinine',
        'diabetes': 'Diabetes',
        'hypertension': 'Hypertension'
    })

    # Drop columns not used as features
    cols_to_drop = ['patient_id', 'diagnosis', 'readmission_30d', 'mortality']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Handle missing values
    df = df.dropna()

    # Calculate Risk Score based on the Framingham Risk Score framework
    # Reference: D'Agostino et al. (2008), "General Cardiovascular Risk Profile"
    # Adapted scoring: weighted sum of normalized risk factors
    risk_score = (
        (df['Age'] / 90) * 25 +
        (df['BMI'] / 40) * 15 +
        (df['Systolic_BP'] / 180) * 20 +
        (df['Cholesterol_Total'] / 300) * 10 +
        (df['Glucose'] / 200) * 10 +
        (df['Creatinine'] / 5) * 5 +
        (df['Diabetes'] * 8) +
        (df['Hypertension'] * 7)
    )

    # Normalize Risk Score to 0-100 scale
    df['Risk_Score'] = ((risk_score - risk_score.min()) / (risk_score.max() - risk_score.min()) * 100).round(1)

    # Assign Risk Levels based on Score thresholds
    conditions = [
        (df['Risk_Score'] < 40),
        (df['Risk_Score'] >= 40) & (df['Risk_Score'] < 70),
        (df['Risk_Score'] >= 70)
    ]
    choices = ['Low', 'Medium', 'High']
    df['Risk_Level'] = np.select(conditions, choices, default='Unknown')
    
    return df

# --- Model Training ---
@st.cache_resource
def train_models(df):
    # Encoding Gender
    le_gender = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

    # Map Risk Level to numerical for classification target
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Risk_Level_Encoded'] = df['Risk_Level'].map(risk_map)

    # Define Features and Targets
    features = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol_Total', 'Glucose', 'Creatinine', 'Diabetes', 'Hypertension', 'Gender_Encoded']
    
    X = df[features]
    y_reg = df['Risk_Score']
    y_clf = df['Risk_Level_Encoded']

    # Train/Test Split (80/20)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear Regression (Risk Score)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_reg_train)

    # Train Logistic Regression (Risk Classification)
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg.fit(X_train_scaled, y_clf_train)

    # --- Evaluation Metrics ---
    y_reg_pred = lin_reg.predict(X_test_scaled)
    y_clf_pred = log_reg.predict(X_test_scaled)

    metrics = {
        'r2': r2_score(y_reg_test, y_reg_pred),
        'mae': mean_absolute_error(y_reg_test, y_reg_pred),
        'accuracy': accuracy_score(y_clf_test, y_clf_pred),
        'conf_matrix': confusion_matrix(y_clf_test, y_clf_pred),
        'class_report': classification_report(y_clf_test, y_clf_pred, target_names=['Low', 'Medium', 'High'], output_dict=True),
        'feature_names': features,
        'coefficients': lin_reg.coef_
    }

    return lin_reg, log_reg, scaler, le_gender, features, metrics

# --- Main Application UI ---

FILE_PATH = 'synthetic_clinical_dataset.csv'
raw_df = load_data(FILE_PATH)

st.markdown("""
<div class="main-header">
    <h1><span class="material-icons">local_hospital</span> MediRisk AI</h1>
    <p>Intelligent Patient Risk Assessment System</p>
</div>
""", unsafe_allow_html=True)

if raw_df is None:
    st.error(f"Dataset file '{FILE_PATH}' not found. Please ensure it is in the same directory.")
else:
    df = preprocess_data(raw_df)
    lin_reg, log_reg, scaler, le_gender, feature_names_used, metrics = train_models(df)

    # --- Compact Input Form (Grid Layout) ---
    st.markdown("""
    <div class="section-header">
        <span class="material-icons">assignment_ind</span> Patient Vitals
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("risk_form"):
        # Row 1: Demographics
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 100, 50)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c3:
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

        # Row 2: Vitals
        c4, c5, c6 = st.columns(3)
        with c4:
            systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
        with c5:
            diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80)
        with c6:
            glucose = st.number_input("Glucose", 50, 300, 100)

        # Row 3: Lab Values & Conditions
        c7, c8 = st.columns(2)
        with c7:
            cholesterol = st.number_input("Cholesterol", 100, 400, 200)
        with c8:
            creatinine = st.number_input("Creatinine", 0.1, 5.0, 1.0, step=0.1)

        # Row 4: Existing Conditions
        c9, c10 = st.columns(2)
        with c9:
            diabetes_input = st.selectbox("Diabetes", ["No", "Yes"])
        with c10:
            hypertension_input = st.selectbox("Hypertension", ["No", "Yes"])

        diabetes = 1 if diabetes_input == "Yes" else 0
        hypertension = 1 if hypertension_input == "Yes" else 0
        
        # Analyze Button
        submitted = st.form_submit_button("CALCULATE RISK ASSESSMENT")

    # --- Prediction Logic ---
    if submitted:
        # Prepare input
        gender_encoded = le_gender.transform([gender])[0]
        input_data = pd.DataFrame([{
            'Age': age,
            'BMI': bmi,
            'Systolic_BP': systolic_bp,
            'Diastolic_BP': diastolic_bp,
            'Cholesterol_Total': cholesterol,
            'Glucose': glucose,
            'Creatinine': creatinine,
            'Diabetes': diabetes,
            'Hypertension': hypertension,
            'Gender_Encoded': gender_encoded
        }])
        
        input_data = input_data[feature_names_used]
        input_scaled = scaler.transform(input_data)
        
        # Predict
        predicted_score = lin_reg.predict(input_scaled)[0]
        predicted_score = np.clip(predicted_score, 0, 100)
        predicted_class_idx = log_reg.predict(input_scaled)[0]
        risk_map_inv = {0: 'Low', 1: 'Medium', 2: 'High'}
        predicted_level = risk_map_inv[predicted_class_idx]
        
        # --- Results Display (Polished) ---
        color_map = {"Low": "#00ff7f", "Medium": "#ffd700", "High": "#ff4500"} 
        badge_color = color_map.get(predicted_level, "grey")
        
        st.markdown(f"""
<div class="result-section">
<div class="section-header" style="justify-content: center;">
<span class="material-icons">analytics</span> Analysis Report
</div>
<div class="result-container">
<div class="result-card">
<h4>RISK SCORE</h4>
<div class="big-number">{predicted_score:.1f}</div>
</div>
<div class="result-card" style="border-bottom: 4px solid {badge_color};">
<h4>RISK LEVEL</h4>
<div class="risk-label" style="color: {badge_color};">{predicted_level.upper()}</div>
</div>
</div>
<div class="notification-box" style="background-color: rgba({int(badge_color[1:3], 16)}, {int(badge_color[3:5], 16)}, {int(badge_color[5:7], 16)}, 0.1); border: 1px solid {badge_color}; color: {badge_color};">
<span class="material-icons">info</span>
{'High Risk: Immediate Medical Attention Recommended' if predicted_level == "High" else 'Moderate Risk: Regular Monitoring Advised' if predicted_level == "Medium" else 'Low Risk: Maintain Healthy Lifestyle'}
</div>
</div>
""", unsafe_allow_html=True)


    # --- Model Performance Section ---
    st.markdown("---")
    st.markdown("""
<div class="section-header">
<span class="material-icons">assessment</span> Model Performance Metrics
</div>
""", unsafe_allow_html=True)
    
    col_reg, col_clf = st.columns(2)
    
    with col_reg:
        st.markdown("##### Linear Regression (Risk Score)")
        st.metric("R² Score", f"{metrics['r2']:.4f}")
        st.metric("Mean Absolute Error", f"{metrics['mae']:.4f}")
    
    with col_clf:
        st.markdown("##### Logistic Regression (Risk Level)")
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        report_df = pd.DataFrame(metrics['class_report']).transpose()
        report_df = report_df.loc[['Low', 'Medium', 'High'], ['precision', 'recall', 'f1-score', 'support']]
        st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))

    # --- Visualizations ---
    st.markdown("---")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("##### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low', 'Medium', 'High'],
                    yticklabels=['Low', 'Medium', 'High'], ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_title('Classification Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig_cm)
    
    with viz_col2:
        st.markdown("##### Feature Importance (Linear Regression Coefficients)")
        feature_labels = ['Age', 'BMI', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'Glucose', 'Creatinine', 'Diabetes', 'Hypertension', 'Gender']
        coefs = metrics['coefficients']
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        
        fig_fi, ax_fi = plt.subplots(figsize=(4, 3))
        colors = ['#00ced1' if c > 0 else '#ff4500' for c in coefs[sorted_idx]]
        ax_fi.barh([feature_labels[i] for i in sorted_idx], np.abs(coefs[sorted_idx]), color=colors)
        ax_fi.set_xlabel('Absolute Coefficient')
        ax_fi.set_title('Feature Importance')
        ax_fi.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_fi)

    # --- Footer ---
    st.markdown("---")
    with st.expander("View Underlying Dataset Stats"):
        st.write(df.describe())
