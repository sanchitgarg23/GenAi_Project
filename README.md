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
    Ensure that `synthetic_clinical_dataset.csv` is present in the project root directory. This file is required for model training.

## Usage

### Running the Application
```bash
streamlit run app.py
```
The application will launch in your default web browser (typically at `http://localhost:8501`).

### Using the Interface
1.  **Patient Vitals**: Enter the patient's information in the "Patient Vitals" section.
    - **Age**: Patient age in years.
    - **Gender**: Biological sex.
    - **BMI**: Body Mass Index.
    - **Systolic BP**: Systolic Blood Pressure (mm Hg).
    - **Diastolic BP**: Diastolic Blood Pressure (mm Hg).
    - **Glucose**: Blood Glucose Level (mg/dL).
    - **Cholesterol**: Total Cholesterol (mg/dL).
    - **Creatinine**: Serum Creatinine Level (mg/dL).
    - **Diabetes**: Diagnosis status (Yes/No).
    - **Hypertension**: Diagnosis status (Yes/No).

2.  **Analyze**: Click the **CALCULATE RISK ASSESSMENT** button.

3.  **View Results**: The "Analysis Report" section will display:
    - **Risk Score**: A numerical value representing the calculated health risk.
    - **Risk Level**: A categorical assessment (Low, Medium, High).
    - **Recommendation**: A brief medical recommendation based on the risk level.

4.  **Model Performance**: Scroll down to view model evaluation metrics, confusion matrix, and feature importance chart.

---

## Technical Implementation

### Data Pipeline
1.  **Data Loading**: Loads clinical data from `synthetic_clinical_dataset.csv` (10,000 patient records).
2.  **Preprocessing**:
    - Drops non-clinical columns (patient_id, diagnosis, readmission_30d, mortality).
    - Renames columns to standard formats.
    - Encodes categorical variables (Gender: Male/Female → 0/1).
    - Handles missing values via `dropna()`.
    - Standardizes numerical features using `StandardScaler`.
3.  **Train/Test Split**: 80/20 split with `random_state=42` for reproducibility.

### Risk Scoring Logic
The system calculates a Risk Score adapted from the **Framingham Risk Score** framework (D'Agostino et al., 2008), a widely-used clinical tool for cardiovascular risk assessment. The adapted formula combines:
- Age (normalized, 25% weight)
- Systolic BP (normalized, 20% weight)
- BMI (normalized, 15% weight)
- Cholesterol and Glucose (normalized, 10% each)
- Creatinine (normalized, 5% weight)
- Diabetes (8%) and Hypertension (7%) penalties

### Machine Learning Models
1.  **Linear Regression**: Predicts continuous Risk Score (0–100).
2.  **Logistic Regression (Multinomial)**: Classifies patients into Low (<40), Medium (40–70), or High (>70) risk levels.

### Project Structure
- `app.py`: Main Streamlit application with UI, data pipeline, model training, metrics display, and prediction logic.
- `requirements.txt`: Python dependencies.
- `synthetic_clinical_dataset.csv`: 10,000-record synthetic clinical dataset.
- `report.tex`: LaTeX project report with methodology and results.