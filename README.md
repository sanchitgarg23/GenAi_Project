# Project 3: Intelligent Patient Risk Assessment & Agentic Health Support System

## From Predictive Healthcare Analytics to Agentic Support

### Project Overview
This project involves the design and implementation of an **AI-based healthcare analytics system** that predicts patient health risks and evolves into an agentic health support assistant.

- **Milestone 1:** Classical machine learning techniques applied to structured clinical patient data to predict health risks such as disease likelihood, using supervised learning models (Linear Regression & Logistic Regression).
- **Milestone 2:** Extension into an agent-based AI application that autonomously reasons about patient risk profiles, retrieves medical guidelines (RAG), and generates structured health summaries with preventive care recommendations.

---

### Constraints & Requirements
- **Team Size:** 3–4 Students
- **API Budget:** Free Tier Only (Open-source models / Free APIs)
- **Framework:** LangGraph (Recommended)
- **Hosting:** Mandatory (Hugging Face Spaces, Streamlit Cloud, or Render)

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **ML Models (M1)** | Logistic Regression, Linear Regression, Scikit-Learn |
| **Agent Framework (M2)** | LangGraph, Chroma/FAISS (RAG) |
| **UI Framework** | Streamlit |
| **LLMs (M2)** | Open-source models or Free-tier APIs |

---

### Milestones & Deliverables

#### Milestone 1: ML-Based Patient Risk Assessment (Mid-Sem)
**Objective:** Design and implement a machine learning–based healthcare analytics system that predicts patient health risks using structured clinical data. This milestone focuses on classical ML workflows *without LLMs*.

**Key Deliverables:**
- Problem understanding & use case description.
- Input–output specification.
- System architecture diagram.
- ML model implementation code.
- Working local application with UI (Streamlit).
- Model performance evaluation report (R², Accuracy, F1, Confusion Matrix).

#### Milestone 2: Agentic AI Health Support Assistant (End-Sem)
**Objective:** Extend the risk assessment system into an agentic AI health support assistant that autonomously reasons over patient data, retrieves medical knowledge, and generates structured health guidance.

**Key Deliverables:**
- **Publicly deployed application** (Link required).
- Agent workflow documentation (States & Nodes).
- Structured health report generation.
- GitHub Repository & Complete Codebase.
- Demo Video (Max 5 mins).

---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | Correct application of ML concepts, Quality of data preprocessing & Feature selection, Model performance & Evaluation metrics, Code modularity & UI usability. |
| **End-Sem** | 30% | Quality & reliability of agent reasoning, Correct RAG implementation & State management, Clarity/Structure of health reports, Ethical responsible AI & Deployment success. |

> [!WARNING]
> Localhost-only demonstrations will **not** be accepted for final submission. Project must be hosted.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/sanchitgarg23/GenAi_Project.git
    cd GenAi_Project
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify Dataset**
    Ensure that `cardio_train.csv` is present in the project root directory. This file is required for model training.

## Usage

### Running the Application
```bash
streamlit run app.py
```
The application will launch in your default web browser (typically at `http://localhost:8501`).

### Using the Interface
1.  **Patient Vitals**: Enter the patient's information in the "Patient Vitals" section.
    - **Age**: Patient age in years.
    - **Gender**: Biological sex (Female/Male).
    - **BMI**: Body Mass Index.
    - **Systolic BP**: Systolic Blood Pressure (mmHg).
    - **Diastolic BP**: Diastolic Blood Pressure (mmHg).
    - **Cholesterol**: Total Cholesterol (mg/dL) — mapped internally to Normal/Above Normal/Well Above Normal.
    - **Glucose**: Blood Glucose Level (mg/dL) — mapped internally to Normal/Above Normal/Well Above Normal.
    - **Smoker**: Smoking status (Yes/No).
    - **Alcohol Consumption**: Alcohol intake status (Yes/No).
    - **Physically Active**: Physical activity status (Yes/No).

2.  **Analyze**: Click the **Predict Risk** button.

3.  **View Results**: The "Analysis Results" section will display:
    - **Cardiovascular Risk Score**: A percentage value representing the predicted cardiovascular disease risk.
    - **Risk Level**: A categorical assessment (Low, Medium, High).
    - **Recommendation**: A brief medical recommendation based on the risk level.

4.  **Model Performance**: Scroll down to view model evaluation metrics, confusion matrix, and feature importance chart.

---

## Technical Implementation

### Dataset
The system uses the **Cardiovascular Disease dataset** (`cardio_train.csv`) containing **70,000 real patient records** from a medical examination study. After preprocessing and outlier removal, approximately **68,600 records** are used for training.

### Data Pipeline
1.  **Data Loading**: Loads cardiovascular data from `cardio_train.csv` (semicolon-separated, 70,000 patient records).
2.  **Preprocessing**:
    - Drops the `id` column (not clinically relevant).
    - Converts age from **days to years** (original dataset stores age in days).
    - Computes **BMI** from height (cm) and weight (kg): `BMI = weight / (height/100)²`.
    - Filters **blood pressure outliers** (Systolic: 70–250 mmHg, Diastolic: 40–150 mmHg, Systolic > Diastolic).
    - Filters **BMI outliers** (valid range: 10–60).
    - Encodes categorical variables (Gender: 1=Female, 2=Male → LabelEncoded).
    - Maps user-entered Cholesterol (mg/dL) → categories: Normal (<200), Above Normal (200–239), Well Above Normal (≥240).
    - Maps user-entered Glucose (mg/dL) → categories: Normal (<100), Above Normal (100–125), Well Above Normal (≥126).
    - Handles missing values via `dropna()`.
    - Standardizes numerical features using `StandardScaler`.
3.  **Train/Test Split**: 80/20 split with `random_state=42` for reproducibility.

### Risk Scoring Logic
The system uses the **Logistic Regression probability output** to calculate cardiovascular disease risk. The model's `predict_proba()` method returns the probability of cardiovascular disease (class 1), which is converted to a percentage risk score:
- **Low Risk** (< 40%): Maintain Healthy Lifestyle
- **Medium Risk** (40–70%): Regular Monitoring Advised
- **High Risk** (> 70%): Immediate Medical Attention Recommended

### Machine Learning Models
1.  **Linear Regression**: Predicts continuous Systolic Blood Pressure values from other patient features (R² ≈ 0.56).
2.  **Logistic Regression**: Binary classification predicting cardiovascular disease risk (Accuracy ≈ 72.7%).
3.  **Decision Tree Classifier**: Alternative classification model for comparison (Accuracy ≈ 73.4%).

### Project Structure
- `app.py`: Main Streamlit application with UI, data pipeline, model training, metrics display, and prediction logic.
- `requirements.txt`: Python dependencies.
- `cardio_train.csv`: 70,000-record cardiovascular disease dataset from medical examination data.
- `report.tex`: LaTeX project report with methodology and results.