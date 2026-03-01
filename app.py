import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# Note: The raw data is executed here during initialization
if __name__ == "__main__":
    DATASET_PATH = 'synthetic_clinical_dataset.csv'
    try:
        raw_clinical_data = load_and_explore_data(DATASET_PATH)
        processed_data, scaled_matrix, feature_scaler, gender_encoder, feature_names = preprocess_features(raw_clinical_data)
        print("Data Pipeline Initialization Complete. Features ready for inference.")
    except Exception as e:
        print(f"Pipeline Initialization Failed: {str(e)}")
