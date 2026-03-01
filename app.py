import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# ========================================================================================
# PAGE CONFIG & STYLING
# ========================================================================================
st.set_page_config(
    page_title="Clinical Risk Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    .stApp { font-family: 'Inter', sans-serif; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    .metric-card h3 { margin: 0; font-size: 14px; font-weight: 500; opacity: 0.7; letter-spacing: 0.5px; text-transform: uppercase; }
    .metric-card .value { font-size: 36px; font-weight: 700; margin: 8px 0 4px; }
    .metric-card .sub { font-size: 13px; opacity: 0.6; }
    
    /* Risk badges */
    .risk-low { color: #22c55e; }
    .risk-moderate { color: #f59e0b; }
    .risk-high { color: #ef4444; }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        padding: 32px 40px;
        border-radius: 20px;
        margin-bottom: 24px;
        border: 1px solid rgba(59,130,246,0.2);
    }
    .app-header h1 {
        margin: 0;
        color: white;
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p { color: #94a3b8; margin: 8px 0 0; font-size: 15px; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ========================================================================================
# CORE: DATA INGESTION & EXPLORATION (with @st.cache_data)
# ========================================================================================
@st.cache_data(show_spinner="Loading clinical dataset...")
def load_and_explore_data(file_path):
    """
    Loads and validates the clinical dataset before ML preprocessing.
    Cached to avoid repeated I/O on Streamlit re-runs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Critical Error: Core clinical dataset is missing from the environment.")
        
    df = pd.read_csv(file_path)
    
    # --- Data Cleaning ---
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
    
    # Drop rows where critical vitals are missing
    df = df.dropna(subset=['age', 'bmi', 'systolic_bp'])
    
    # Standardize column names
    df = df.rename(columns={
        'age': 'Age',
        'sex': 'Gender',
        'bmi': 'BMI',
        'systolic_bp': 'Systolic_BP',
        'diastolic_bp': 'Diastolic_BP',
        'cholesterol': 'Cholesterol_Total',
        'glucose': 'Glucose',
        'diabetes': 'Diabetes',
        'hypertension': 'Hypertension',
        'diagnosis': 'Diagnosis',
        'readmission_30d': 'Readmission_30d',
        'mortality': 'Mortality'
    })
    
    # Remove non-predictive columns
    columns_to_drop = ['patient_id', 'record_date', 'Unnamed: 0']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    return df


# ========================================================================================
# CORE: WEIGHTED CLINICAL RISK SCORING ALGORITHM
# ========================================================================================

# Clinical weight configuration (sums to 1.0)
RISK_WEIGHTS = {
    'bmi':          0.20,
    'systolic_bp':  0.20,
    'diastolic_bp': 0.10,
    'glucose':      0.15,
    'cholesterol':  0.15,
    'age':          0.10,
    'diabetes':     0.05,
    'hypertension': 0.05,
}

# Clinical thresholds for sub-score normalization
CLINICAL_THRESHOLDS = {
    'bmi': {'low': 18.5, 'high': 30.0, 'critical': 40.0},
    'systolic_bp': {'normal': 120.0, 'elevated': 140.0, 'critical': 180.0},
    'diastolic_bp': {'normal': 80.0, 'elevated': 90.0, 'critical': 120.0},
    'glucose': {'low': 70.0, 'normal': 100.0, 'high': 140.0, 'critical': 200.0},
    'cholesterol': {'desirable': 200.0, 'borderline': 240.0, 'high': 300.0},
    'age': {'young': 30, 'middle': 50, 'senior': 65, 'elderly': 80},
}


@st.cache_data(show_spinner="Computing weighted risk scores...")
def compute_weighted_risk_scores(_df):
    """
    Vectorized weighted clinical risk scoring algorithm.
    Each vital sign is normalized to a 0-1 sub-score, weighted, and aggregated.
    Final score is on a 0-100 scale.
    
    Uses NumPy vectorized operations for maximum backend efficiency.
    """
    df = _df.copy()
    
    # --- BMI Sub-Score (0-1) ---
    # Risk increases as BMI deviates from healthy range (18.5-25)
    bmi = df['BMI'].values
    bmi_score = np.where(
        bmi < 18.5,
        np.clip((18.5 - bmi) / 8.0, 0, 1),  # underweight risk
        np.where(
            bmi <= 25.0,
            0.0,  # healthy range = no risk
            np.where(
                bmi <= 30.0,
                (bmi - 25.0) / 5.0 * 0.5,  # overweight = moderate
                np.clip(0.5 + (bmi - 30.0) / 20.0 * 0.5, 0.5, 1.0)  # obese = high
            )
        )
    )
    
    # --- Systolic BP Sub-Score (0-1) ---
    sbp = df['Systolic_BP'].values
    sbp_score = np.where(
        sbp < 120, 0.0,
        np.where(
            sbp <= 140,
            (sbp - 120.0) / 20.0 * 0.5,
            np.clip(0.5 + (sbp - 140.0) / 40.0 * 0.5, 0.5, 1.0)
        )
    )
    
    # --- Diastolic BP Sub-Score (0-1) ---
    dbp = df['Diastolic_BP'].values
    dbp_score = np.where(
        dbp < 80, 0.0,
        np.where(
            dbp <= 90,
            (dbp - 80.0) / 10.0 * 0.5,
            np.clip(0.5 + (dbp - 90.0) / 30.0 * 0.5, 0.5, 1.0)
        )
    )
    
    # --- Glucose Sub-Score (0-1) ---
    glu = df['Glucose'].values
    glu_score = np.where(
        glu < 70,
        np.clip((70 - glu) / 30.0, 0, 0.5),  # hypoglycemia risk
        np.where(
            glu <= 100, 0.0,  # normal
            np.where(
                glu <= 140,
                (glu - 100.0) / 40.0 * 0.5,  # pre-diabetic
                np.clip(0.5 + (glu - 140.0) / 60.0 * 0.5, 0.5, 1.0)  # diabetic
            )
        )
    )
    
    # --- Cholesterol Sub-Score (0-1) ---
    chol = df['Cholesterol_Total'].values
    chol_score = np.where(
        chol < 200, 0.0,
        np.where(
            chol <= 240,
            (chol - 200.0) / 40.0 * 0.5,
            np.clip(0.5 + (chol - 240.0) / 60.0 * 0.5, 0.5, 1.0)
        )
    )
    
    # --- Age Sub-Score (0-1) ---
    age = df['Age'].values
    age_score = np.where(
        age < 30, 0.0,
        np.where(
            age <= 50, (age - 30.0) / 20.0 * 0.3,
            np.where(
                age <= 65, 0.3 + (age - 50.0) / 15.0 * 0.3,
                np.clip(0.6 + (age - 65.0) / 25.0 * 0.4, 0.6, 1.0)
            )
        )
    )
    
    # --- Binary Flag Sub-Scores ---
    diabetes_score = df['Diabetes'].values.astype(float)
    hypertension_score = df['Hypertension'].values.astype(float)
    
    # --- Weighted Aggregation ---
    weighted_score = (
        RISK_WEIGHTS['bmi'] * bmi_score +
        RISK_WEIGHTS['systolic_bp'] * sbp_score +
        RISK_WEIGHTS['diastolic_bp'] * dbp_score +
        RISK_WEIGHTS['glucose'] * glu_score +
        RISK_WEIGHTS['cholesterol'] * chol_score +
        RISK_WEIGHTS['age'] * age_score +
        RISK_WEIGHTS['diabetes'] * diabetes_score +
        RISK_WEIGHTS['hypertension'] * hypertension_score
    )
    
    # Scale to 0-100
    risk_score = np.clip(weighted_score * 100, 0, 100)
    
    # Assign to dataframe
    df['Risk_Score'] = np.round(risk_score, 2)
    df['Risk_Category'] = pd.cut(
        df['Risk_Score'],
        bins=[-1, 30, 60, 100],
        labels=['Low', 'Moderate', 'High']
    )
    
    # Store individual sub-scores for breakdown
    df['BMI_SubScore'] = np.round(bmi_score * 100, 1)
    df['SBP_SubScore'] = np.round(sbp_score * 100, 1)
    df['DBP_SubScore'] = np.round(dbp_score * 100, 1)
    df['Glucose_SubScore'] = np.round(glu_score * 100, 1)
    df['Cholesterol_SubScore'] = np.round(chol_score * 100, 1)
    df['Age_SubScore'] = np.round(age_score * 100, 1)
    
    return df


def compute_single_patient_risk(age, gender, bmi, sbp, dbp, glucose, cholesterol, diabetes, hypertension):
    """
    Computes weighted risk score for a single patient input.
    Returns (score, category, sub_scores_dict).
    """
    # BMI
    if bmi < 18.5:
        bmi_s = min((18.5 - bmi) / 8.0, 1.0)
    elif bmi <= 25.0:
        bmi_s = 0.0
    elif bmi <= 30.0:
        bmi_s = (bmi - 25.0) / 5.0 * 0.5
    else:
        bmi_s = min(0.5 + (bmi - 30.0) / 20.0 * 0.5, 1.0)
    
    # Systolic BP
    if sbp < 120:
        sbp_s = 0.0
    elif sbp <= 140:
        sbp_s = (sbp - 120.0) / 20.0 * 0.5
    else:
        sbp_s = min(0.5 + (sbp - 140.0) / 40.0 * 0.5, 1.0)
    
    # Diastolic BP
    if dbp < 80:
        dbp_s = 0.0
    elif dbp <= 90:
        dbp_s = (dbp - 80.0) / 10.0 * 0.5
    else:
        dbp_s = min(0.5 + (dbp - 90.0) / 30.0 * 0.5, 1.0)
    
    # Glucose
    if glucose < 70:
        glu_s = min((70 - glucose) / 30.0, 0.5)
    elif glucose <= 100:
        glu_s = 0.0
    elif glucose <= 140:
        glu_s = (glucose - 100.0) / 40.0 * 0.5
    else:
        glu_s = min(0.5 + (glucose - 140.0) / 60.0 * 0.5, 1.0)
    
    # Cholesterol
    if cholesterol < 200:
        chol_s = 0.0
    elif cholesterol <= 240:
        chol_s = (cholesterol - 200.0) / 40.0 * 0.5
    else:
        chol_s = min(0.5 + (cholesterol - 240.0) / 60.0 * 0.5, 1.0)
    
    # Age
    if age < 30:
        age_s = 0.0
    elif age <= 50:
        age_s = (age - 30.0) / 20.0 * 0.3
    elif age <= 65:
        age_s = 0.3 + (age - 50.0) / 15.0 * 0.3
    else:
        age_s = min(0.6 + (age - 65.0) / 25.0 * 0.4, 1.0)
    
    weighted = (
        RISK_WEIGHTS['bmi'] * bmi_s +
        RISK_WEIGHTS['systolic_bp'] * sbp_s +
        RISK_WEIGHTS['diastolic_bp'] * dbp_s +
        RISK_WEIGHTS['glucose'] * glu_s +
        RISK_WEIGHTS['cholesterol'] * chol_s +
        RISK_WEIGHTS['age'] * age_s +
        RISK_WEIGHTS['diabetes'] * float(diabetes) +
        RISK_WEIGHTS['hypertension'] * float(hypertension)
    )
    
    score = round(min(max(weighted * 100, 0), 100), 2)
    category = 'Low' if score <= 30 else ('Moderate' if score <= 60 else 'High')
    
    sub_scores = {
        'BMI': round(bmi_s * 100, 1),
        'Systolic BP': round(sbp_s * 100, 1),
        'Diastolic BP': round(dbp_s * 100, 1),
        'Glucose': round(glu_s * 100, 1),
        'Cholesterol': round(chol_s * 100, 1),
        'Age': round(age_s * 100, 1),
        'Diabetes': round(float(diabetes) * 100, 1),
        'Hypertension': round(float(hypertension) * 100, 1),
    }
    
    return score, category, sub_scores


# ========================================================================================
# FEATURE ENGINEERING PIPELINE
# ========================================================================================
@st.cache_data(show_spinner="Engineering features...")
def preprocess_features(_df):
    """
    Constructs the final feature matrix and applies scaling for ML.
    Cached for backend efficiency.
    """
    df = _df.copy()
    
    le_gender = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

    features = [
        'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
        'Cholesterol_Total', 'Glucose', 'Diabetes', 
        'Hypertension', 'Gender_Encoded'
    ]
    
    X = df[features]
    y = df['Readmission_30d']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, y, scaler, le_gender, features


# ========================================================================================
# MODEL TRAINING PIPELINE
# ========================================================================================
@st.cache_resource(show_spinner="Training clinical model...")
def train_clinical_model(_X_scaled, _y_target):
    """
    Trains Logistic Regression for readmission_30d prediction.
    Cached with @st.cache_resource to persist model across re-runs.
    Returns model + evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        _X_scaled, _y_target, test_size=0.2, random_state=42
    )

    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    y_prob = log_reg.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = None

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }

    return log_reg, metrics


# ========================================================================================
# STREAMLIT DASHBOARD UI
# ========================================================================================
def render_header():
    st.markdown("""
    <div class="app-header">
        <h1>🏥 Clinical Risk Intelligence Dashboard</h1>
        <p>Weighted clinical risk scoring algorithm with ML-powered 30-day readmission prediction</p>
    </div>
    """, unsafe_allow_html=True)


def render_overview_metrics(df):
    """Renders the top-level KPI cards."""
    total = len(df)
    high_risk = len(df[df['Risk_Category'] == 'High'])
    moderate_risk = len(df[df['Risk_Category'] == 'Moderate'])
    low_risk = len(df[df['Risk_Category'] == 'Low'])
    avg_score = df['Risk_Score'].mean()
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Patients</h3>
            <div class="value">{total:,}</div>
            <div class="sub">In dataset</div>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Risk Score</h3>
            <div class="value">{avg_score:.1f}</div>
            <div class="sub">Out of 100</div>
        </div>""", unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔴 High Risk</h3>
            <div class="value risk-high">{high_risk:,}</div>
            <div class="sub">{high_risk/total*100:.1f}% of patients</div>
        </div>""", unsafe_allow_html=True)
    
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🟡 Moderate Risk</h3>
            <div class="value risk-moderate">{moderate_risk:,}</div>
            <div class="sub">{moderate_risk/total*100:.1f}% of patients</div>
        </div>""", unsafe_allow_html=True)
    
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🟢 Low Risk</h3>
            <div class="value risk-low">{low_risk:,}</div>
            <div class="sub">{low_risk/total*100:.1f}% of patients</div>
        </div>""", unsafe_allow_html=True)


def render_dashboard_tab(df):
    """Risk distribution charts and analytics."""
    st.markdown("### 📊 Risk Score Distribution")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Histogram of risk scores using pd.cut for binning
        bins = pd.cut(df['Risk_Score'], bins=20)
        hist_counts = bins.value_counts().sort_index()
        hist_counts.index = [f"{i.left:.0f}-{i.right:.0f}" for i in hist_counts.index]
        st.bar_chart(hist_counts, use_container_width=True)
    
    with col2:
        # Risk category breakdown
        st.markdown("#### Risk Category Breakdown")
        cat_counts = df['Risk_Category'].value_counts()
        st.bar_chart(cat_counts, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔬 Risk Factor Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Average Sub-Scores by Risk Category**")
        sub_cols = ['BMI_SubScore', 'SBP_SubScore', 'DBP_SubScore', 
                    'Glucose_SubScore', 'Cholesterol_SubScore', 'Age_SubScore']
        avg_by_cat = df.groupby('Risk_Category')[sub_cols].mean()
        avg_by_cat.columns = ['BMI', 'Systolic BP', 'Diastolic BP', 'Glucose', 'Cholesterol', 'Age']
        st.dataframe(avg_by_cat.round(1), use_container_width=True)
    
    with col2:
        st.markdown("**Diagnosis Distribution by Risk**")
        diag_risk = pd.crosstab(df['Diagnosis'], df['Risk_Category'])
        st.dataframe(diag_risk, use_container_width=True)
    
    with col3:
        st.markdown("**Readmission Rate by Risk Category**")
        readmit = df.groupby('Risk_Category')['Readmission_30d'].mean() * 100
        readmit_df = pd.DataFrame({'Readmission Rate (%)': readmit.round(2)})
        st.dataframe(readmit_df, use_container_width=True)
        st.bar_chart(readmit, use_container_width=True)


def render_patient_table_tab(df):
    """Filterable, color-coded patient risk table."""
    st.markdown("### 📋 Patient Risk Assessment Table")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Filter by Risk Category", ['Low', 'Moderate', 'High'], default=['Low', 'Moderate', 'High'])
    with col2:
        age_range = st.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    with col3:
        diag_filter = st.multiselect("Filter by Diagnosis", df['Diagnosis'].unique().tolist(), default=df['Diagnosis'].unique().tolist())
    
    # Apply filters
    filtered = df[
        (df['Risk_Category'].isin(risk_filter)) &
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Diagnosis'].isin(diag_filter))
    ]
    
    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} patients**")
    
    # Display table with key columns
    display_cols = ['Age', 'Gender', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
                    'Glucose', 'Cholesterol_Total', 'Diabetes', 'Hypertension',
                    'Diagnosis', 'Risk_Score', 'Risk_Category', 'Readmission_30d']
    
    # Color-code the risk scores
    def highlight_risk(val):
        if val == 'High':
            return 'background-color: #fee2e2; color: #dc2626; font-weight: 600'
        elif val == 'Moderate':
            return 'background-color: #fef3c7; color: #d97706; font-weight: 600'
        else:
            return 'background-color: #dcfce7; color: #16a34a; font-weight: 600'
    
    styled = filtered[display_cols].head(200).style.map(
        highlight_risk, subset=['Risk_Category']
    )
    
    st.dataframe(styled, use_container_width=True, height=500)


def render_model_tab(metrics):
    """Model performance metrics display."""
    st.markdown("### 🤖 ML Model Performance (Logistic Regression)")
    st.markdown("*Predicts 30-day hospital readmission using patient clinical features*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col4:
        roc_val = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
        st.metric("ROC-AUC", roc_val)
    
    st.markdown("---")
    st.markdown("#### Confusion Matrix")
    cm = metrics['confusion_matrix']
    cm_df = pd.DataFrame(cm, index=['Actual: No', 'Actual: Yes'], columns=['Predicted: No', 'Predicted: Yes'])
    st.dataframe(cm_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Weight Configuration")
    st.markdown("These are the clinical weights used in the risk scoring algorithm:")
    weights_df = pd.DataFrame({
        'Risk Factor': list(RISK_WEIGHTS.keys()),
        'Weight': list(RISK_WEIGHTS.values()),
        'Contribution (%)': [w * 100 for w in RISK_WEIGHTS.values()]
    })
    st.dataframe(weights_df, use_container_width=True)


def render_prediction_tab(model, scaler, le_gender, feature_names):
    """Individual patient risk prediction form."""
    st.markdown("### 🩺 Individual Patient Risk Assessment")
    st.markdown("*Enter patient vitals to compute their weighted risk score and ML-based readmission probability*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=55, step=1)
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5, step=0.1)
    
    with col2:
        sbp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=130, step=1)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=85, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=30.0, max_value=400.0, value=110.0, step=0.1)
    
    with col3:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80.0, max_value=400.0, value=210.0, step=0.1)
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    
    if st.button("⚡ Compute Risk Score", type="primary", use_container_width=True):
        # Weighted risk score
        score, category, sub_scores = compute_single_patient_risk(
            age, gender, bmi, sbp, dbp, glucose, cholesterol, diabetes, hypertension
        )
        
        # ML prediction
        gender_encoded = 1 if gender == 'Male' else 0
        patient_features = np.array([[age, bmi, sbp, dbp, cholesterol, glucose, diabetes, hypertension, gender_encoded]])
        patient_scaled = scaler.transform(patient_features)
        readmission_prob = model.predict_proba(patient_scaled)[0][1]
        readmission_pred = model.predict(patient_scaled)[0]
        
        st.markdown("---")
        
        # Results
        col_a, col_b = st.columns(2)
        
        with col_a:
            risk_color = '#ef4444' if category == 'High' else ('#f59e0b' if category == 'Moderate' else '#22c55e')
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3>Weighted Risk Score</h3>
                <div class="value" style="color: {risk_color}; font-size: 48px;">{score}</div>
                <div class="sub" style="font-size: 18px; color: {risk_color}; font-weight: 600;">{category} Risk</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            readmit_color = '#ef4444' if readmission_pred == 1 else '#22c55e'
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3>ML Readmission Prediction</h3>
                <div class="value" style="color: {readmit_color}; font-size: 48px;">{readmission_prob*100:.1f}%</div>
                <div class="sub" style="font-size: 18px; color: {readmit_color}; font-weight: 600;">{'Likely to be Readmitted' if readmission_pred == 1 else 'Unlikely to be Readmitted'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 📊 Risk Factor Breakdown")
        breakdown_df = pd.DataFrame({
            'Factor': list(sub_scores.keys()),
            'Sub-Score (0-100)': list(sub_scores.values()),
            'Weight': [RISK_WEIGHTS[k.lower().replace(' ', '_')] if k.lower().replace(' ', '_') in RISK_WEIGHTS else RISK_WEIGHTS.get(k.lower().replace(' bp', '_bp'), 0) for k in sub_scores.keys()],
        })
        # Fix weight mapping
        weight_map = {
            'BMI': 0.20, 'Systolic BP': 0.20, 'Diastolic BP': 0.10,
            'Glucose': 0.15, 'Cholesterol': 0.15, 'Age': 0.10,
            'Diabetes': 0.05, 'Hypertension': 0.05
        }
        breakdown_df['Weight'] = breakdown_df['Factor'].map(weight_map)
        breakdown_df['Weighted Contribution'] = (breakdown_df['Sub-Score (0-100)'] * breakdown_df['Weight']).round(2)
        
        st.dataframe(breakdown_df, use_container_width=True)
        st.bar_chart(breakdown_df.set_index('Factor')['Sub-Score (0-100)'], use_container_width=True)


# ========================================================================================
# MAIN APP
# ========================================================================================
def main():
    render_header()
    
    DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_clinical_dataset.csv')
    
    try:
        # Pipeline: Load -> Score -> Engineer -> Train
        raw_data = load_and_explore_data(DATASET_PATH)
        scored_data = compute_weighted_risk_scores(raw_data)
        processed_data, X_scaled, y_target, scaler, le_gender, feature_names = preprocess_features(raw_data)
        model, metrics = train_clinical_model(X_scaled, y_target.values)
        
        # Overview KPIs
        render_overview_metrics(scored_data)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard", 
            "📋 Patient Risk Table", 
            "🤖 Model Performance",
            "🩺 Individual Prediction"
        ])
        
        with tab1:
            render_dashboard_tab(scored_data)
        
        with tab2:
            render_patient_table_tab(scored_data)
        
        with tab3:
            render_model_tab(metrics)
        
        with tab4:
            render_prediction_tab(model, scaler, le_gender, feature_names)
    
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        st.error(f"❌ Pipeline Error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
