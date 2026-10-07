# AvertCare: Project Walkthrough & Roadmap

This document outlines everything we have built for the **Cognizant Hackathon (Use Case #6: Hospital Readmission Risk Prediction)**, our architectural decisions, and the exact roadmap for the final M1 model handoff.

---

## 🏗️ What We Have Built So Far (The Full Stack)

We have successfully transitioned from a 0-to-1 scaffold to a fully functional, loosely coupled Prescriptive Clinical Decision Support system. The architecture is feature-flagged, allowing seamless toggling between deterministic mock data and live intelligent inference.

### 1. Vector Database Seeding & RAG Feature Extraction (Track 1)
- **Automated Pipeline (`scripts/seed_rag_engine.py`)**: An ETL script that ingests tabular patient data (`diabetic_data.csv`), generates realistic unstructured clinical notes (injecting random Social Determinants of Health), and embeds them using `sentence-transformers/all-MiniLM-L6-v2`.
- **Qdrant Upsert & $RAG_{risk}$**: The script connects to the local Qdrant Vector DB, upserts the embeddings along with metadata (e.g., historical interventions), and calculates a localized readmission rate ($RAG_{risk}$) by querying the top-5 nearest neighbors for each patient. 
- **M1 Handoff**: The script automatically exports `data/processed/train_with_rag.csv`, providing our Data Scientist (M1) with the augmented dataset needed to train the final XGBoost/Transformer models.

### 2. The Backend API & Live RAG Retrieval (Track 2)
- **FastAPI Framework**: Exposes `POST /api/predict` and `POST /api/twin-patients` using strict Pydantic v2 schemas mapped to FHIR standards.
- **Live Qdrant Integration (`services.py`)**: The `run_rag_retrieval` service now actively embeds incoming clinical notes, queries the live Qdrant container for the top-3 most similar past patient encounters, and computes the semantic neighborhood risk index on the fly. (Controlled via `LIVE_RAG_ENABLED`).
- **Prescriptive LLM Care Plan (`services.py`)**: Integrated the `google-genai` SDK. When `LIVE_LLM_ENABLED=true`, the backend passes the patient's risk score, extracted SDoH barriers, top SHAP drivers, and successful twin-patient interventions to the Gemini 2.0 Flash model, which synthesizes a structured, prioritized 3-point clinical discharge plan and calculates estimated CMS penalty savings.

### 3. Clinician Authentication & Security (Track 3)
- **Firebase Auth Middleware (`app/core/auth.py`)**: Implemented a FastAPI dependency (`require_auth`) that validates Firebase JWT ID tokens on all protected routes using `firebase-admin`.
- **Zero-Friction Dev Bypass**: When `FIREBASE_PROJECT_ID` is not set in the local `.env`, the middleware acts as a transparent passthrough, allowing UI development to proceed without requiring live authentication tokens.

### 4. MLOps & Observability Stack (Track 4)
- **Multi-Container Orchestration (`docker-compose.yml`)**: Now orchestrates FastAPI, Next.js, Qdrant, Redis, Prometheus, and Grafana.
- **Prometheus Scraper**: Connects to the FastAPI `/metrics` endpoint to aggregate time-series telemetry.
- **Auto-Provisioned Grafana**: A customized Grafana dashboard (`grafana/provisioning/dashboards/avertcare.json`) is automatically loaded on boot (available at `http://localhost:3001`). It visualizes critical IT Ops metrics:
  - API Request Throughput (RPS)
  - P95 / P99 Inference Latency
  - HTTP 2xx Success vs. 4xx/5xx Error Rates

### 5. The Clinical Dashboard (Next.js)
- **Tech Stack**: Next.js 15 (Node 20), Tailwind CSS, and `next-themes`.
- **Dark Mode Aesthetic**: A strict high-contrast dark-mode UI designed to reduce alert fatigue, mimicking modern Epic EHR interfaces without relying on heavy interactive CLI installers.
- **Dynamic Render**: Asynchronously fetches from the FastAPI backend to gracefully render the Risk Score Gauge, SDoH warning badges, the Twin-Patient RAG cohort table, a CSS-based TreeSHAP force-plot breakdown, and the GenAI prescriptive care plan.

---

## 🚀 How to Run the System

1. **Spin up Infrastructure:**
   ```bash
   docker-compose up -d --build
   ```
2. **Seed the Vector Database (Unblocks M1):**
   ```bash
   pip install -r scripts/requirements-seed.txt
   python scripts/seed_rag_engine.py --csv data/raw/diabetic_data.csv
   ```
3. **Enable Live Features:** Add your `GEMINI_API_KEY` to `backend/.env` and set `LIVE_RAG_ENABLED=true` and `LIVE_LLM_ENABLED=true`.

---

## 🎯 Next Steps (Final M1 Handoff)
1. **Load the Final Models**: M1 will drop the trained XGBoost `.pkl` and Bio_ClinicalBERT artifacts into the `ml_engine/models/` directory (which is mounted as a Docker volume).
2. **Wire Inference**: Inside `backend/app/services.py`, we will replace the `run_prediction` mock logic placeholder with the actual `joblib.load()` and `xgboost` inference calls.
3. **Frontend Auth Hookup**: Add the Firebase Auth login modal to the Next.js UI to generate and pass the Bearer tokens to the backend in production.
