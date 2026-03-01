import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# --- Data Ingestion & Exploration ---
@st.cache_data
def load_and_explore_data(file_path):
    """
    Loads and validates the clinical dataset before ML preprocessing.
    """
    print(f"Loading clinical dataset from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError("Critical Error: Core clinical dataset is missing from the environment.")
        
    df = pd.read_csv(file_path)
    
    # --- Comprehensive Exploratory Data Analysis (EDA) ---
  
    
    # 1. check the first few rows

    print(df.head())
    print("\n")
    
    # 2. Check data types and overall info
    df.info()
    print("\n")
    
    # 3. Statistical summary of the data

    print(df.describe().T)
    print("\n")
    
    # 4. Check for missing values across the dataset

    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0] if missing_data.sum() > 0 else "No missing values detected.")
    print("\n")
    
    # 5. Check for duplicate records

    duplicates = df.duplicated().sum()
    print(f"Total duplicate records found: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"-> Removed {duplicates} duplicate records.")

    
    # --- Data Cleaning & Consistency Check ---
    # Drop rows where critical vitals are absolutely required but missing
    initial_rows = len(df)
    df = df.dropna(subset=['age', 'bmi', 'systolic_bp'])
    if len(df) < initial_rows:
        print(f"Data Cleaning: Removed {initial_rows - len(df)} records with missing critical vitals.")
    
    # Standardize nomenclature for downstream processing
    df = df.rename(columns={
        'age': 'Age',
        'sex': 'Gender',
        'bmi': 'BMI',
        'systolic_bp': 'Systolic_BP',
        'diastolic_bp': 'Diastolic_BP',
        'cholesterol': 'Cholesterol_Total',
        'glucose': 'Glucose',
        'diabetes': 'Diabetes',
        'hypertension': 'Hypertension'
    })
    
    # Remove surrogate IDs and timestamps which hold no predictive power
    columns_to_drop = ['patient_id', 'record_date', 'Unnamed: 0']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    return df

# --- Feature Engineering Pipeline ---
def preprocess_features(df):
    """
    Constructs the final feature matrix and applies scaling for Machine Learning algorithms.
    """
    print(f"Engineering features for {len(df)} patient records...")
    
    # 1. Encode Categorical Variables (Gender -> binary state)
    le_gender = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

    # 2. Define the exact feature geometry expected by the models
    features = [
        'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
        'Cholesterol_Total', 'Glucose', 'Diabetes', 
        'Hypertension', 'Gender_Encoded'
    ]
    
    X = df[features]

    # 3. Apply standard scaling (Z-score normalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler, le_gender, features

def generate_synthetic_targets(df):
    """
    Calculates the Synthetic Risk Score acting as Ground Truth for the models.
    """
    risk_score = (
        (df['Age'] / 90) * 30 +  
        (df['BMI'] / 40) * 20 + 
        (df['Systolic_BP'] / 180) * 15 +
        (df['Cholesterol_Total'] / 300) * 10 +
        (df['Glucose'] / 200) * 10 +
        (df['Diabetes'] * 5) +
        (df['Hypertension'] * 10)
    )

    df['Risk_Score'] = ((risk_score - risk_score.min()) / (risk_score.max() - risk_score.min()) * 100).round(1)

    conditions = [
        (df['Risk_Score'] < 40),
        (df['Risk_Score'] >= 40) & (df['Risk_Score'] < 70),
        (df['Risk_Score'] >= 70)
    ]
    df['Risk_Level'] = np.select(conditions, ['Low', 'Medium', 'High'], default='Unknown')
    
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Risk_Level_Encoded'] = df['Risk_Level'].map(risk_map)
    
    return df['Risk_Score'], df['Risk_Level_Encoded']

# --- Model Training Pipeline ---
@st.cache_resource
def train_risk_models(X_scaled, y_score, y_level):
    """
    Trains dual Machine Learning models for predicting both exact risk score and discrete risk tier.
    Validates models to ensure clinical viability.
    """
    print(f"\nTraining Risk Models on {len(X_scaled)} records...")
    
    X_train, X_test, y_score_train, y_score_test, y_level_train, y_level_test = train_test_split(
        X_scaled, y_score, y_level, test_size=0.2, random_state=42
    )

    # 1. Linear Regression for precise 0-100 Synthetic Risk Score
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_score_train)
    y_score_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_score_test, y_score_pred)
    print(f"Linear Regression target score (0-100): Trained and Validated. MSE: {mse:.2f}")

    # 2. Logistic Regression for Low/Medium/High Classification
    # Max iterations raised for convergence on clinical data
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_level_train)
    y_level_pred = log_reg.predict(X_test)
    acc = accuracy_score(y_level_test, y_level_pred)
    print(f"Logistic Regression Categorical Model (Low/Medium/High): Trained and Validated. Accuracy: {acc:.2f}")

    return lin_reg, log_reg

# Note: The raw data is executed here during initialization
if __name__ == "__main__":
    DATASET_PATH = 'synthetic_clinical_dataset.csv'
    try:
        raw_clinical_data = load_and_explore_data(DATASET_PATH)
        processed_data, scaled_matrix, feature_scaler, gender_encoder, feature_names = preprocess_features(raw_clinical_data)
        
        scores, levels = generate_synthetic_targets(processed_data)
        lin_model, log_model = train_risk_models(scaled_matrix, scores, levels)
        
        print("Data Pipeline Initialization Complete. Models ready for inference.")
    except Exception as e:
        print(f"Pipeline Initialization Failed: {str(e)}")
