# AvertCare

## AI-Powered 30-Day Hospital Readmission Risk Prediction & Prevention

AvertCare is a prescriptive Clinical Decision Support (CDS) system designed to predict a patient's risk of unplanned 30-day hospital readmission and provide actionable insights to help clinicians reduce that risk.

Unlike traditional systems that only provide a risk score, AvertCare combines structured patient data, clinical notes, NLP-based Social Determinants of Health (SDoH) extraction, machine learning, Retrieval-Augmented Generation (RAG), explainable AI, and Generative AI.

### Core Idea

> **Predict the risk → Explain the risk → Retrieve similar cases → Recommend an intervention**

---

## Problem

Hospital readmission prediction models commonly rely on structured patient information such as demographics, diagnoses, laboratory results, vitals, comorbidities, and previous admissions.

However, important risk factors can be hidden in unstructured clinical notes, including:

- Financial difficulties
- Transportation barriers
- Living alone
- Medication access problems
- Other Social Determinants of Health

A risk score alone does not explain why a patient is at risk or what action could reduce that risk.

---

## Solution

AvertCare combines structured patient information, clinical text, and historical patient encounters into a single prediction and decision-support pipeline.

```text
Patient Data
     │
     ├── Structured Data ────────┐
     │                           │
     └── Clinical Notes          │
             │                   │
             ▼                   │
       NLP / SDoH Extraction     │
             │                   │
             └─────────┬─────────┘
                       ▼
                Feature Engineering
                       │
              ┌────────┴────────┐
              ▼                 ▼
        ML Risk Model      Twin-Patient RAG
              │                 │
              └────────┬────────┘
                       ▼
                Hybrid Prediction
                       │
                       ▼
                  Explainable AI
                       │
                       ▼
                GenAI Copilot
                       │
                       ▼
             Personalized Care Plan
```

---

## Key Features

### 1. 30-Day Readmission Prediction

Predicts the probability of an unplanned hospital readmission within 30 days using patient-level clinical features.

### 2. Clinical NLP & SDoH Extraction

Processes unstructured clinical notes to identify relevant risk factors such as financial distress, transportation barriers, living conditions, and medication-related issues.

### 3. Twin-Patient RAG

Retrieves semantically similar historical patient encounters from a vector database.

Retrieved cases provide information about:

- Similar patient profiles
- Readmission outcomes
- Risk factors
- Interventions applied to similar patients

These retrieval results are used as additional evidence for prediction and recommendation.

### 4. Hybrid ML Model

Combines:

```text
Structured Patient Features
        +
SDoH Features
        +
RAG-derived Features
        ↓
Hybrid Risk Model
```

The proposed architecture uses XGBoost for risk prediction.

### 5. Explainable AI

TreeSHAP is used to identify the features contributing to the model's prediction, providing a clearer explanation of why a patient is classified as high-risk.

### 6. Prescriptive GenAI Copilot

Uses patient risk factors and retrieved historical cases to generate personalized intervention recommendations.

Example:

```text
High Readmission Risk

Recommended Actions:
1. Schedule telehealth follow-up within 72 hours.
2. Perform medication reconciliation.
3. Arrange transportation assistance.
```

### 7. Financial Impact Estimation

Provides an estimate of potential CMS penalty avoidance associated with preventing readmissions.

---

## Architecture

```text
┌─────────────────────────────┐
│     Clinical Dashboard      │
│       Next.js / React       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       FastAPI Backend       │
│     FHIR / API Validation   │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌────────────────┐
│ Tabular Data │  │ RAG Retrieval  │
│              │  │                │
│ Patient Data │  │ Embeddings     │
│ Vitals       │  │ Vector DB      │
│ Labs         │  │ Twin Patients  │
│ Comorbidities│  │                │
└──────┬───────┘  └───────┬────────┘
       │                  │
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │ Hybrid ML + XAI  │
       │                  │
       │ XGBoost          │
       │ SDoH Features    │
       │ RAG Features     │
       │ TreeSHAP         │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │  GenAI Copilot   │
       │                  │
       │ Risk Explanation │
       │ Recommendations  │
       │ Care Plan        │
       └──────────────────┘
```

---

## ML Pipeline

1. **Data Processing** — Clean and transform structured patient data.
2. **Clinical NLP** — Extract relevant SDoH information from clinical notes.
3. **Embedding & Retrieval** — Generate clinical embeddings and retrieve similar historical encounters.
4. **Risk Prediction** — Combine structured, NLP-derived, and RAG-derived features.
5. **Explainability** — Use TreeSHAP to identify important prediction factors.
6. **Recommendation** — Generate an intervention plan using the prediction and retrieved evidence.

---

## Model Evaluation

The architecture is evaluated incrementally:

| Model | Features |
|---|---|
| Baseline | Tabular features |
| Enhanced | Tabular + SDoH features |
| Hybrid | Tabular + SDoH + RAG features |

### Metrics

- ROC-AUC
- F1 Score
- Precision
- Recall
- Inference Latency

This comparison measures the contribution of clinical text and RAG features over a conventional tabular baseline.

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Next.js, React |
| UI | Tailwind CSS, shadcn/ui |
| Backend | FastAPI, Python |
| Machine Learning | XGBoost |
| NLP | ClinicalBERT / Bio_ClinicalBERT |
| RAG | Embeddings + Vector Search |
| Vector Database | FAISS / Qdrant |
| Explainability | TreeSHAP |
| Generative AI | LLM |
| Authentication | Firebase |
| Data Processing | Polars |
| Interoperability | HL7 FHIR |
| Monitoring | Prometheus, Grafana |
| Experiment Tracking | MLflow / W&B |

---

## Project Structure

```text
AvertCare/
│
├── frontend/
├── backend/
├── ml/
├── rag/
├── monitoring/
├── notebooks/
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm
- Git
- Docker
- Required API credentials and service configuration

### Clone

```bash
git clone <YOUR_REPOSITORY_URL>
cd AvertCare
```

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Environment variables should be configured according to the services used by the project.

---

## MLOps & Monitoring

The architecture includes Prometheus and Grafana for monitoring:

- Model inference latency
- RAG retrieval latency
- API throughput
- HTTP errors
- Resource utilization

---

## Key Differentiators

- **Predictive → Prescriptive:** Goes beyond a risk score to recommend actions.
- **Twin-Patient RAG:** Uses similar historical cases as evidence.
- **Multimodal:** Combines structured patient data with clinical text.
- **Explainable:** Uses TreeSHAP for prediction explanations.
- **FHIR-Compatible:** Designed around healthcare interoperability standards.
- **MLOps-Oriented:** Includes monitoring and observability.

---

## Future Scope

- Real-world EHR integration
- SMART on FHIR deployment
- Real-time patient monitoring
- Advanced clinical NLP
- Model drift detection
- Bias and fairness monitoring
- Continuous model retraining
- Human-in-the-loop clinical workflows
- Longitudinal patient risk modeling

---

## Team

Developed for the **Cognizant Hackathon — Use Case #6: Hospital Readmission Risk Prediction**.

**Team Members:**

- Add Team Member
- Add Team Member
- Add Team Member
- Add Team Member

---

## Disclaimer

AvertCare is an academic/hackathon prototype intended for demonstration and research purposes.

It is not intended to provide medical diagnosis, treatment, or autonomous clinical decisions. Predictions and recommendations should be reviewed by qualified healthcare professionals before real-world clinical use.
'''
