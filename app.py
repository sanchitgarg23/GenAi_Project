import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

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
        'hypertension': 'Hypertension',
        'diagnosis': 'Diagnosis',
        'readmission_30d': 'Readmission_30d',
        'mortality': 'Mortality'
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

    # 2. Define the exact feature geometry expected by the models (Excluding target variables and non-predictive categorical col like Diagnosis for now)
    features = [
        'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
        'Cholesterol_Total', 'Glucose', 'Diabetes', 
        'Hypertension', 'Gender_Encoded'
    ]
    
    X = df[features]
    
    # Target variable choice: Readmission_30d
    y = df['Readmission_30d']

    # 3. Apply standard scaling (Z-score normalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, y, scaler, le_gender, features

# --- Model Training Pipeline ---
@st.cache_resource
def train_clinical_model(X_scaled, y_target):
    """
    Trains a Logistic Regression model to predict readmission_30d.
    Validates model to ensure clinical viability using standard classification metrics.
    """
    print(f"\nTraining Risk Model on {len(X_scaled)} records for Readmission Outcome...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_target, test_size=0.2, random_state=42
    )

    # 1. Logistic Regression for Binary Classification
    # using class_weight='balanced' to handle potential class imbalances
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    y_prob = log_reg.predict_proba(X_test)[:, 1]

    # Evaluate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = "N/A (Only one class present in test set)"

    print("\n--- Model Evaluation (Readmission_30d) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return log_reg

# Note: The raw data is executed here during initialization
if __name__ == "__main__":
    DATASET_PATH = 'synthetic_clinical_dataset.csv'
    try:
        raw_clinical_data = load_and_explore_data(DATASET_PATH)
        processed_data, scaled_matrix, target_variable, feature_scaler, gender_encoder, feature_names = preprocess_features(raw_clinical_data)
        
        # Train model specifically for readmission prediction
        readmission_model = train_clinical_model(scaled_matrix, target_variable)
        
        print("\nData Pipeline Initialization Complete. Model ready for inference.")
    except Exception as e:
        print(f"Pipeline Initialization Failed: {str(e)}")
