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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── CSS Custom Properties: Medical Dark Theme ── */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: #1a1f2e;
        --bg-card-hover: #222838;
        --border-subtle: rgba(255,255,255,0.06);
        --border-accent: rgba(14,165,233,0.25);
        --accent-teal: #0ea5e9;
        --accent-blue: #3b82f6;
        --accent-indigo: #6366f1;
        --accent-green: #22c55e;
        --accent-yellow: #f59e0b;
        --accent-red: #ef4444;
        --text-primary: #f1f5f9;
        --text-muted: #94a3b8;
        --text-dim: #64748b;
        --glass-bg: rgba(17,24,39,0.7);
        --glass-border: rgba(255,255,255,0.08);
        --radius-lg: 16px;
        --radius-md: 12px;
        --shadow-card: 0 4px 24px rgba(0,0,0,0.35);
        --shadow-glow-teal: 0 0 20px rgba(14,165,233,0.15);
        --shadow-glow-red: 0 0 20px rgba(239,68,68,0.2);
        --shadow-glow-green: 0 0 20px rgba(34,197,94,0.15);
        --shadow-glow-yellow: 0 0 20px rgba(245,158,11,0.15);
    }

    /* ── Keyframe Animations ── */
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    @keyframes float-in {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes border-shimmer {
        0% { border-color: rgba(14,165,233,0.2); }
        50% { border-color: rgba(14,165,233,0.5); }
        100% { border-color: rgba(14,165,233,0.2); }
    }
    @keyframes critical-pulse {
        0%, 100% { box-shadow: 0 0 8px rgba(239,68,68,0.3); }
        50% { box-shadow: 0 0 24px rgba(239,68,68,0.6); }
    }
    @keyframes status-dot {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.4); opacity: 0.5; }
    }

    /* ── Global App Background ── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(170deg, var(--bg-primary) 0%, #0d1321 40%, var(--bg-secondary) 100%) !important;
        color: var(--text-primary);
    }
    .stApp > header { background: transparent !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* ── Metric Cards (Glass Morphism) ── */
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, rgba(26,31,46,0.85) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: var(--radius-lg);
        padding: 24px 28px;
        color: var(--text-primary);
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-card);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.3s ease;
        animation: float-in 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue));
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
        opacity: 0.7;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-card), var(--shadow-glow-teal);
        border-color: var(--border-accent);
    }
    .metric-card .card-icon {
        font-size: 28px;
        margin-bottom: 8px;
        display: block;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 12px;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 36px;
        font-weight: 800;
        margin: 8px 0 4px;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 13px;
        color: var(--text-dim);
        font-weight: 400;
    }

    /* ── Risk Color Classes ── */
    .risk-low { color: var(--accent-green) !important; }
    .risk-moderate { color: var(--accent-yellow) !important; }
    .risk-high { color: var(--accent-red) !important; }
    .glow-green { border-color: rgba(34,197,94,0.3) !important; box-shadow: var(--shadow-card), var(--shadow-glow-green) !important; }
    .glow-yellow { border-color: rgba(245,158,11,0.3) !important; box-shadow: var(--shadow-card), var(--shadow-glow-yellow) !important; }
    .glow-red { border-color: rgba(239,68,68,0.3) !important; box-shadow: var(--shadow-card), var(--shadow-glow-red) !important; }

    /* ── App Header ── */
    .app-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #0c1a2e 30%, #122040 60%, #0a0e1a 100%);
        padding: 36px 44px;
        border-radius: 20px;
        margin-bottom: 28px;
        border: 1px solid var(--border-accent);
        position: relative;
        overflow: hidden;
        animation: float-in 0.6s ease-out;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(14,165,233,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .app-header::after {
        content: '';
        position: absolute;
        bottom: -40%; left: -10%;
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .app-header .header-row {
        display: flex;
        align-items: center;
        gap: 16px;
        position: relative;
        z-index: 1;
    }
    .app-header .header-icon {
        font-size: 38px;
        filter: drop-shadow(0 0 8px rgba(14,165,233,0.4));
    }
    .app-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #38bdf8 40%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .app-header .subtitle {
        color: var(--text-muted);
        margin: 6px 0 0 0;
        font-size: 15px;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.25);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 12px;
        color: var(--accent-green);
        font-weight: 600;
        margin-top: 12px;
        position: relative;
        z-index: 1;
    }
    .status-dot {
        width: 7px; height: 7px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: status-dot 2s ease-in-out infinite;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 6px;
        border: 1px solid var(--glass-border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 22px;
        font-weight: 500;
        color: var(--text-muted) !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text-primary) !important; background: rgba(255,255,255,0.04); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(59,130,246,0.15)) !important;
        color: var(--accent-teal) !important;
        font-weight: 600;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue)) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(14,165,233,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(14,165,233,0.4) !important;
    }

    /* ── Inputs, Selects, Sliders ── */
    .stTextInput > div > div, .stNumberInput > div > div,
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: var(--bg-card) !important;
        border-color: var(--glass-border) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-md) !important;
    }
    .stSlider > div > div > div { background: var(--accent-teal) !important; }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
    }

    /* ── Section Dividers ── */
    hr { border-color: var(--border-subtle) !important; }
    .stMarkdown h3, .stMarkdown h4 { color: var(--text-primary) !important; }

    /* ── Risk Result Card (Prediction) ── */
    .risk-result-card {
        background: linear-gradient(145deg, var(--bg-card), rgba(26,31,46,0.9));
        backdrop-filter: blur(12px);
        border-radius: var(--radius-lg);
        padding: 32px;
        text-align: center;
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-card);
        position: relative;
        overflow: hidden;
        animation: float-in 0.5s ease-out;
    }
    .risk-result-card .score-ring {
        width: 140px; height: 140px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 16px;
        font-size: 42px;
        font-weight: 800;
        position: relative;
    }
    .risk-result-card .result-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 8px;
    }
    .risk-result-card .result-category {
        font-size: 20px;
        font-weight: 700;
        margin-top: 4px;
    }
    .risk-tier-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-top: 12px;
    }
    .tier-low { background: rgba(34,197,94,0.12); color: var(--accent-green); border: 1px solid rgba(34,197,94,0.3); }
    .tier-moderate { background: rgba(245,158,11,0.12); color: var(--accent-yellow); border: 1px solid rgba(245,158,11,0.3); }
    .tier-high { background: rgba(239,68,68,0.12); color: var(--accent-red); border: 1px solid rgba(239,68,68,0.3); }

    /* ── Floating Alert Badge ── */
    .alert-badge {
        position: absolute;
        top: 12px; right: 12px;
        background: var(--accent-red);
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 12px;
        animation: critical-pulse 1.5s ease-in-out infinite;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Sub-Score Progress Bars ── */
    .subscore-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .subscore-label {
        width: 110px;
        font-size: 13px;
        font-weight: 500;
        color: var(--text-muted);
        text-align: right;
        flex-shrink: 0;
    }
    .subscore-bar-bg {
        flex: 1;
        height: 8px;
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        overflow: hidden;
    }
    .subscore-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }
    .subscore-val {
        width: 45px;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
        text-align: left;
        flex-shrink: 0;
    }

    /* ── Vitals grid card accent overrides ── */
    .metric-card.card-total::before { background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue)); }
    .metric-card.card-avg::before { background: linear-gradient(90deg, var(--accent-blue), var(--accent-indigo)); }
    .metric-card.card-high::before { background: var(--accent-red); }
    .metric-card.card-moderate::before { background: var(--accent-yellow); }
    .metric-card.card-low::before { background: var(--accent-green); }
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
    
    # Use custom threshold to improve Precision & Accuracy
    y_prob = log_reg.predict_proba(X_test)[:, 1]
    CUSTOM_THRESHOLD = 0.65
    y_pred = (y_prob >= CUSTOM_THRESHOLD).astype(int)

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
        <div class="header-row">
            <span class="header-icon">🏥</span>
            <h1>Clinical Risk Intelligence Dashboard</h1>
        </div>
        <p class="subtitle">Weighted clinical risk scoring algorithm with ML-powered 30-day readmission prediction</p>
        <div class="status-badge">
            <span class="status-dot"></span>
            System Online — Real-time Analysis Active
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_overview_metrics(df):
    """Renders the top-level vitals grid KPI cards with icons and color-coded glows."""
    total = len(df)
    high_risk = len(df[df['Risk_Category'] == 'High'])
    moderate_risk = len(df[df['Risk_Category'] == 'Moderate'])
    low_risk = len(df[df['Risk_Category'] == 'Low'])
    avg_score = df['Risk_Score'].mean()
    avg_age = df['Age'].mean()
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card card-total">
            <span class="card-icon">🩺</span>
            <h3>Total Patients</h3>
            <div class="value">{total:,}</div>
            <div class="sub">Avg age {avg_age:.0f} yrs</div>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        score_class = 'risk-high' if avg_score > 60 else ('risk-moderate' if avg_score > 30 else 'risk-low')
        st.markdown(f"""
        <div class="metric-card card-avg">
            <span class="card-icon">📊</span>
            <h3>Avg Risk Score</h3>
            <div class="value {score_class}">{avg_score:.1f}</div>
            <div class="sub">Weighted composite</div>
        </div>""", unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class="metric-card card-high glow-red">
            <span class="card-icon">🔴</span>
            <h3>Critical / High</h3>
            <div class="value risk-high">{high_risk:,}</div>
            <div class="sub">{high_risk/total*100:.1f}% — Immediate review</div>
        </div>""", unsafe_allow_html=True)
    
    with c4:
        st.markdown(f"""
        <div class="metric-card card-moderate glow-yellow">
            <span class="card-icon">🟡</span>
            <h3>Elevated</h3>
            <div class="value risk-moderate">{moderate_risk:,}</div>
            <div class="sub">{moderate_risk/total*100:.1f}% — Monitor closely</div>
        </div>""", unsafe_allow_html=True)
    
    with c5:
        st.markdown(f"""
        <div class="metric-card card-low glow-green">
            <span class="card-icon">🟢</span>
            <h3>Normal</h3>
            <div class="value risk-low">{low_risk:,}</div>
            <div class="sub">{low_risk/total*100:.1f}% — Within range</div>
        </div>""", unsafe_allow_html=True)


def render_dashboard_tab(df):
    """Risk distribution charts and analytics."""
    st.markdown("### Risk Score Distribution")
    
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
    st.markdown("### Risk Factor Analysis")
    
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
    st.markdown("### Patient Risk Assessment Table")
    
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
    """Model performance metrics display with themed cards."""
    st.markdown("### ML Model Performance (Logistic Regression)")
    st.markdown("*Predicts 30-day hospital readmission using patient clinical features*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    model_metrics = [
        ("Accuracy", f"{metrics['accuracy']:.4f}", "🎯", col1),
        ("Precision", f"{metrics['precision']:.4f}", "🔬", col2),
        ("Recall", f"{metrics['recall']:.4f}", "📡", col3),
        ("ROC-AUC", f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A", "📈", col4),
    ]
    
    for label, val, icon, col in model_metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <span class="card-icon">{icon}</span>
                <h3>{label}</h3>
                <div class="value" style="font-size:28px;">{val}</div>
            </div>""", unsafe_allow_html=True)
    
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
    st.markdown("### Individual Patient Risk Assessment")
    st.markdown("*Enter patient vitals to compute their weighted risk score and ML-based readmission probability*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", min_value=0, max_value=120, value=55, step=1)
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=27.5, step=0.1)
    
    with col2:
        sbp = st.slider("Systolic BP (mmHg)", min_value=70, max_value=250, value=130, step=1)
        dbp = st.slider("Diastolic BP (mmHg)", min_value=40, max_value=150, value=85, step=1)
        glucose = st.slider("Glucose (mg/dL)", min_value=30.0, max_value=400.0, value=110.0, step=0.1)
    
    with col3:
        cholesterol = st.slider("Cholesterol (mg/dL)", min_value=80.0, max_value=400.0, value=210.0, step=0.1)
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    
    if st.button("Calculate", type="primary", use_container_width=True):
        # Weighted risk score
        score, category, sub_scores = compute_single_patient_risk(
            age, gender, bmi, sbp, dbp, glucose, cholesterol, diabetes, hypertension
        )
        
        # ML prediction
        gender_encoded = 1 if gender == 'Male' else 0
        patient_features = np.array([[age, bmi, sbp, dbp, cholesterol, glucose, diabetes, hypertension, gender_encoded]])
        patient_scaled = scaler.transform(patient_features)
        readmission_prob = model.predict_proba(patient_scaled)[0][1]
        
        # Apply custom threshold for prediction
        CUSTOM_THRESHOLD = 0.65
        readmission_pred = 1 if readmission_prob >= CUSTOM_THRESHOLD else 0
        
        st.session_state['prediction_results'] = {
            'score': score,
            'category': category,
            'sub_scores': sub_scores,
            'readmission_prob': readmission_prob,
            'readmission_pred': readmission_pred
        }
        
    if 'prediction_results' in st.session_state:
        res = st.session_state['prediction_results']
        
        st.markdown("---")
        
        # --- Dynamic Risk Result Cards ---
        risk_color = '#ef4444' if res['category'] == 'High' else ('#f59e0b' if res['category'] == 'Moderate' else '#22c55e')
        tier_class = 'tier-high' if res['category'] == 'High' else ('tier-moderate' if res['category'] == 'Moderate' else 'tier-low')
        glow_class = 'glow-red' if res['category'] == 'High' else ('glow-yellow' if res['category'] == 'Moderate' else 'glow-green')
        ring_bg = f'radial-gradient(circle, rgba({"239,68,68" if res["category"]=="High" else ("245,158,11" if res["category"]=="Moderate" else "34,197,94")},0.12) 0%, transparent 70%)'
        alert_html = '<span class="alert-badge">⚠ Critical</span>' if res['category'] == 'High' else ''
        tier_label = 'Critical' if res['category'] == 'High' else ('Elevated' if res['category'] == 'Moderate' else 'Normal')
        
        readmit_color = '#ef4444' if res['readmission_pred'] == 1 else '#22c55e'
        readmit_glow = 'glow-red' if res['readmission_pred'] == 1 else 'glow-green'
        readmit_alert = '<span class="alert-badge">⚠ Alert</span>' if res['readmission_pred'] == 1 else ''
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            <div class="risk-result-card {glow_class}">
                {alert_html}
                <div class="result-label">Weighted Risk Score</div>
                <div class="score-ring" style="background: {ring_bg}; border: 3px solid {risk_color};">
                    <span style="color: {risk_color};">{res['score']}</span>
                </div>
                <div class="result-category" style="color: {risk_color};">{res['category']} Risk</div>
                <div class="risk-tier-badge {tier_class}">● {tier_label}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            readmit_ring_bg = f'radial-gradient(circle, rgba({"239,68,68" if res["readmission_pred"]==1 else "34,197,94"},0.12) 0%, transparent 70%)'
            st.markdown(f"""
            <div class="risk-result-card {readmit_glow}">
                {readmit_alert}
                <div class="result-label">ML Readmission Prediction</div>
                <div class="score-ring" style="background: {readmit_ring_bg}; border: 3px solid {readmit_color};">
                    <span style="color: {readmit_color};">{res['readmission_prob']*100:.1f}%</span>
                </div>
                <div class="result-category" style="color: {readmit_color};">{'Likely to be Readmitted' if res['readmission_pred'] == 1 else 'Unlikely to be Readmitted'}</div>
                <div class="risk-tier-badge {'tier-high' if res['readmission_pred'] == 1 else 'tier-low'}">● {'High Risk' if res['readmission_pred'] == 1 else 'Low Risk'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### Risk Factor Breakdown")
        
        # Visual sub-score progress bars
        weight_map = {
            'BMI': 0.20, 'Systolic BP': 0.20, 'Diastolic BP': 0.10,
            'Glucose': 0.15, 'Cholesterol': 0.15, 'Age': 0.10,
            'Diabetes': 0.05, 'Hypertension': 0.05
        }
        
        bars_html = '<div style="margin-top:12px;">'
        for factor, value in res['sub_scores'].items():
            bar_color = '#ef4444' if value > 70 else ('#f59e0b' if value > 35 else '#22c55e')
            bars_html += f'''
            <div class="subscore-row">
                <span class="subscore-label">{factor}</span>
                <div class="subscore-bar-bg">
                    <div class="subscore-bar-fill" style="width:{min(value, 100)}%; background:{bar_color};"></div>
                </div>
                <span class="subscore-val" style="color:{bar_color};">{value}</span>
            </div>'''
        bars_html += '</div>'
        st.markdown(bars_html, unsafe_allow_html=True)
        
        # Also keep the data table for detail
        st.markdown("<br>", unsafe_allow_html=True)
        breakdown_df = pd.DataFrame({
            'Factor': list(res['sub_scores'].keys()),
            'Sub-Score (0-100)': list(res['sub_scores'].values()),
        })
        breakdown_df['Weight'] = breakdown_df['Factor'].map(weight_map)
        breakdown_df['Weighted Contribution'] = (breakdown_df['Sub-Score (0-100)'] * breakdown_df['Weight']).round(2)
        
        st.dataframe(breakdown_df, use_container_width=True)


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
            "Dashboard", 
            "Patient Risk Table", 
            "Model Performance",
            "Individual Prediction"
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
        st.error(f"{str(e)}")
    except Exception as e:
        st.error(f"Pipeline Error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
