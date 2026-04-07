import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    mean_squared_error, 
    r2_score, 
    confusion_matrix, 
    classification_report, 
    mean_absolute_error
)

# setting up page
st.set_page_config(
    page_title="MediRisk | Patient Assessment",
    page_icon="🏥",
    layout="centered"
)

# function to load data
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

@st.cache_data
def preprocess_data(df):
    # renaming columns so they look better and are easier to use
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

    # drop columns that we dont need for the ML models
    cols_to_drop = ['patient_id', 'diagnosis', 'readmission_30d', 'mortality']
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=[c])

    # handle missing values for data preprocessing (taught in class)
    df = df.dropna()

    # calculating a risk score based on some clinical framework 
    # reference from class or framingham score
    risk_score = (
        (df['Age'] / 90) * 25 +(df['BMI'] / 40) * 15 +(df['Systolic_BP'] / 180) * 20 +(df['Cholesterol_Total'] / 300) * 10 +(df['Glucose'] / 200) * 10 +(df['Creatinine'] / 5) * 5 +(df['Diabetes']*8)+(df['Hypertension'] * 7)
    )

    # normalizing to scale 0-100
    df['Risk_Score'] = ((risk_score - risk_score.min()) / (risk_score.max() - risk_score.min()) * 100).round(1)

    # setting up labels for supervised learning classification task
    conditions = [
        (df['Risk_Score'] < 40),
        (df['Risk_Score'] >= 40) & (df['Risk_Score'] < 70),
        (df['Risk_Score'] >= 70)
    ]
    choices = ['Low', 'Medium', 'High']
    df['Risk_Level'] = np.select(conditions, choices, default='Unknown')
    
    return df

@st.cache_resource
def train_models(df):
    # label encoding as taught in preprocessing
    le_gender = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

    # risk level mapped to numbers for multiclass logistic regression
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Risk_Level_Encoded'] = df['Risk_Level'].map(risk_map)

    # features and targets
    features = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol_Total', 'Glucose', 'Creatinine', 'Diabetes', 'Hypertension', 'Gender_Encoded']
    
    X = df[features]
    y_reg = df['Risk_Score']
    y_clf = df['Risk_Level_Encoded']

    # train test splitting 80/20
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # feature scaling using standard scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_reg_train)

    # Train logistic regression
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg.fit(X_train_scaled, y_clf_train)
    
    # Train Decision Tree (added based on syllabus)
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_clf.fit(X_train_scaled, y_clf_train)

    # model evaluation on test set
    y_reg_pred = lin_reg.predict(X_test_scaled)
    y_clf_pred = log_reg.predict(X_test_scaled)
    y_dt_pred = dt_clf.predict(X_test_scaled)

    # compute mse and rmse
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)

    metrics = {
        'r2': r2_score(y_reg_test, y_reg_pred),
        'mae': mean_absolute_error(y_reg_test, y_reg_pred),
        'mse': mse,
        'rmse': rmse,
        'accuracy': accuracy_score(y_clf_test, y_clf_pred),
        'dt_accuracy': accuracy_score(y_clf_test, y_dt_pred),
        'conf_matrix': confusion_matrix(y_clf_test, y_clf_pred),
        'class_report': classification_report(y_clf_test, y_clf_pred, target_names=['Low', 'Medium', 'High'], output_dict=True),
        'feature_names': features,
        'coefficients': lin_reg.coef_
    }

    return lin_reg, log_reg, dt_clf, scaler, le_gender, features, metrics

# UI components
FILE_PATH = 'synthetic_clinical_dataset.csv'
raw_df = load_data(FILE_PATH)

st.title("🏥 MediRisk AI - Patient Assessment")

if raw_df is None:
    st.error(f"Dataset file '{FILE_PATH}' not found. Check if the file is there.")
else:
    df = preprocess_data(raw_df)
    lin_reg, log_reg, dt_clf, scaler, le_gender, feature_names_used, metrics = train_models(df)

    st.subheader("Enter Patient Vitals")
    
    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 100, 50)
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
            systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
            cholesterol = st.number_input("Cholesterol", 100, 400, 200)
            diabetes_input = st.selectbox("Diabetes", ["No", "Yes"])
            
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80)
            glucose = st.number_input("Glucose", 50, 300, 100)
            creatinine = st.number_input("Creatinine", 0.1, 5.0, 1.0, step=0.1)
            hypertension_input = st.selectbox("Hypertension", ["No", "Yes"])

        diabetes = 1 if diabetes_input == "Yes" else 0
        hypertension = 1 if hypertension_input == "Yes" else 0
        
        submitted = st.form_submit_button("Predict Risk")

    if submitted:
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
        
        predicted_score = lin_reg.predict(input_scaled)[0]
        predicted_score = np.clip(predicted_score, 0, 100)
        
        predicted_class_idx = log_reg.predict(input_scaled)[0]
        risk_map_inv = {0: 'Low', 1: 'Medium', 2: 'High'}
        predicted_level = risk_map_inv[predicted_class_idx]
        
        st.subheader("Analysis Results")
        st.write(f"**Predicted Risk Score:** {predicted_score:.1f}")
        
        if predicted_level == "Low":
            st.success(f"**Predicted Risk Level:** {predicted_level} Risk - Maintain Healthy Lifestyle")
        elif predicted_level == "Medium":
            st.warning(f"**Predicted Risk Level:** {predicted_level} Risk - Regular Monitoring Advised")
        else:
            st.error(f"**Predicted Risk Level:** {predicted_level} Risk - Immediate Medical Attention Recommended")


    st.markdown("---")
    st.subheader("Model Performance")
    
    col_reg, col_clf = st.columns(2)
    
    with col_reg:
        st.markdown("##### Linear Regression")
        st.write(f"**R² Score:** {metrics['r2']:.4f}")
        st.write(f"**MAE:** {metrics['mae']:.4f}")
        st.write(f"**MSE:** {metrics['mse']:.4f}")
        st.write(f"**RMSE:** {metrics['rmse']:.4f}")
    
    with col_clf:
        st.markdown("##### Logistic Regression")
        st.write(f"**Accuracy:** {metrics['accuracy']*100:.2f}%")
        st.markdown("##### Decision Tree")
        st.write(f"**Accuracy:** {metrics['dt_accuracy']*100:.2f}%")
        
        report_df = pd.DataFrame(metrics['class_report']).transpose()
        report_df = report_df.loc[['Low', 'Medium', 'High'], ['precision', 'recall', 'f1-score', 'support']]
        st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))

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
        plt.tight_layout()
        st.pyplot(fig_cm)
    
    with viz_col2:
        st.markdown("##### Feature Importance")
        feature_labels = ['Age', 'BMI', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'Glucose', 'Creatinine', 'Diabetes', 'Hypertension', 'Gender']
        coefs = metrics['coefficients']
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        
        fig_fi, ax_fi = plt.subplots(figsize=(4, 3))
        colors = ['#00ced1' if c > 0 else '#ff4500' for c in coefs[sorted_idx]]
        ax_fi.barh([feature_labels[i] for i in sorted_idx], np.abs(coefs[sorted_idx]), color=colors)
        ax_fi.set_xlabel('Coefficient')
        ax_fi.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_fi)
