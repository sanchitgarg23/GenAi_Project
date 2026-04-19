# Final Project Report: MediRisk AI
**Intelligent Patient Risk Assessment & Agentic Health Support System**

---

## 1. Abstract
Cardiovascular and metabolic diseases are leading global health challenges. Early detection and risk assessment are critical for prevention and effective management. This project presents **MediRisk AI**, a sophisticated, two-phase healthcare platform. 

In **Phase 1**, the system acts as a machine learning-based application designed to predict patient health risks using structured clinical data. Utilizing a dataset of 10,000 patient records, we trained Logistic Regression and Decision Tree models to predict and classify risk levels. In **Phase 2**, the system extends into a completely autonomous **Agentic AI Health Assistant**. Powered by LangGraph, FAISS Vector Search, and Llama-3 (via ChatGroq), the system reasons over the patient’s risk profile, retrieves verified medical guidelines via Retrieval-Augmented Generation (RAG), and generates personalized, structured medical reports. 

The entire solution is deployed on a professional Streamlit web interface to ensure clinical accessibility.

---

## 2. Introduction
### 2.1 Background
The increasing prevalence of lifestyle-related diseases such as hypertension, diabetes, and heart disease places a significant burden on healthcare systems. While numerical risk stratification predicts *if* someone is at risk, patients and doctors often need actionable context on *what to do next*. 

### 2.2 Problem Statement
There is a need for an integrated system that can instantaneously process patient vitals, predict health risk, and immediately transition into providing personalized, verified clinical action plans without the risk of AI hallucination.

### 2.3 Objectives
- To develop a machine learning model capable of high-accuracy risk assessment.
- To use an agentic state-graph framework (LangGraph) for memory-safe conversational UI.
- To implement Retrieval-Augmented Generation (RAG) to ensure the LLM strictly adheres to authorized medical guidelines.
- To deploy a robust public-facing application summarizing patient health.

---

## 3. Phase 1: Machine Learning & Risk Assessment

### 3.1 Data Preparation
A dataset comprising 10,000 patient records was used. The features included Age, Gender, BMI, Systolic/Diastolic BP, Cholesterol, Glucose, Creatinine, Diabetes, and Hypertension.
- **Preprocessing:** Records were cleansed of null values (`dropna`). Categorical fields (e.g., gender) were transformed via `LabelEncoder`.
- **Scaling:** Due to vast differences in feature distributions (e.g., Blood Glucose vs Creatinine), standard scaling (`StandardScaler`) was applied to ensure the model weighted all metrics properly based on variance.

### 3.2 Machine Learning Models
Two distinct supervised learning models were trained using `scikit-learn`:
1. **Logistic Regression:** Used primarily to extract a continuous probability estimate (`predict_proba()`) that directly translates into the patient's exact 0-100% Risk Score. Logistic regression was selected because it naturally outputs probabilities and provides highly interpretable Feature Importance coefficients (vital for medical AI transparency). 
2. **Decision Tree Classifier:** Implemented as a secondary model for highly interpretable node-splitting classification.

### 3.3 Risk Level Classification
Based on the probability scores, patients are instantly categorized into:
- **Low Risk (< 40%)**
- **Medium Risk (40-70%)**
- **High Risk (> 70%)**

---

## 4. Phase 2: Agentic AI Health Assistant

### 4.1 System Architecture (LangGraph)
Rather than executing a single, linear prompt, the system employs **LangGraph** to construct a sophisticated state machine composed of nodes.
- **AgentState:** A memory dictionary that continuously carries the patient's specific vitals, their numerical risk score, retrieved medical text, and full chat history across the AI's execution steps.
- **Nodes & Edges:** The graph connects a strict pipeline: `Retrieve -> Reason_and_Generate -> End`.

### 4.2 Retrieval-Augmented Generation (RAG)
To guarantee the AI does not hallucinate medical facts, we utilized **FAISS (Facebook AI Similarity Search)**.
1. Authoritative text (`medical_guidelines.txt`) was split into bite-sized document chunks using `CharacterTextSplitter`.
2. These chunks were converted into mathematical matrices via **HuggingFace Embeddings**.
3. When the user initiates a query, FAISS retrieves the top `k=2` most mathematically similar clinical guidelines and injects them directly into the AgentState.

### 4.3 Large Language Model (Llama-3 via Groq)
The generative reasoning is executed using the `ChatGroq` API, specifically implementing the `llama-3.1-8b-instant` model. 
- **Prompt Engineering:** The LLM receives a strictly engineered system prompt demanding it: 
  *(A)* Read the patient's generated risk profile.
  *(B)* Read the FAISS retrieved guidelines.
  *(C)* Output a highly structured, empathetic diagnosis report strictly attributing facts to the guidelines while rendering a clinical disclaimer.

---

## 5. System Robustness & UI

### 5.1 Error Handling
Given the unpredictability of Third-Party LLM APIs, our architecture invokes the final LangGraph graph via `try...except` exception blocks. If the RAG lookup fails or API quota exceeds, the system gracefully degrades by popping the chat block and issuing an `st.error()` message, fully preventing a system crash on the frontend.

### 5.2 User Interface
Built using **Streamlit**, the application features:
- **Phase 1 Tab:** Input sliders and categorical dropdowns coupled with a high-visibility, color-coded Glassmorphism risk meter.
- **Phase 2 Tab:** A fluid, conversational chat interface allowing the user to organically speak to their medical data.
- **Telemetry Tab:** A detailed breakdown displaying the model's Confusion Matrix, ROC-AUC curve, Classification Report, and Feature Importance graphs.

---

## 6. Conclusion
The **MediRisk AI** project fulfills its goal of transitioning from static analytics to dynamic, agentic support. By fusing Logistic Regression with FAISS vector search and Llama-3, the system demonstrates how classical machine learning workflows can be successfully integrated into modern Agentic architectures. The implementation highlights the necessity of strict state management (LangGraph) and context-control (RAG) when deploying Large Language Models in healthcare settings.
