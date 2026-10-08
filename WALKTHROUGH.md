# AvertCare: Project Walkthrough & Roadmap

This document outlines everything we have built for the **Cognizant Hackathon (Use Case #6: Hospital Readmission Risk Prediction)**, our architectural decisions, and the current system status.

---

## 🏗️ What We Have Built (The Full Stack)

We have successfully built a fully functional, loosely coupled **Prescriptive Clinical Decision Support System** with 7 containerised services, an end-to-end ML pipeline, and a clinician-facing dark-mode dashboard.

---

### 1. Vector Database Seeding & RAG Feature Extraction
- **Automated Pipeline (`scripts/seed_rag_engine.py`)**: Ingests tabular patient data (`diabetic_data.csv`), generates unstructured clinical notes (injecting Social Determinants of Health), and embeds them using `sentence-transformers/all-MiniLM-L6-v2`.
- **Qdrant Upsert & RAGrisk**: Connects to the local Qdrant Vector DB, upserts embeddings along with metadata, and calculates a localised readmission rate by querying the top-5 nearest neighbours.
- **M1 Handoff**: Automatically exports `data/processed/train_with_rag.csv` — the augmented dataset used to train the final models.

### 2. ML Engine — Phase 4–6 (M1's Work)
- **RAG Ablation Study** (`ml_engine/run_phase4_6_rag.py`): Trained and evaluated 6 models (Logistic Regression, Random Forest, FT-Transformer) × (Tabular Only, Tabular + RAG) on the augmented dataset.
- **Winning Model**: **Random Forest + RAG (`RF_RAG`)** — AUROC 0.6971 vs. 0.6725 Tabular-only (+0.0246 from RAG).
- **SHAP Explainability**: Global and local SHAP explanations generated; stored in `ml_engine/shap_output/`.
- **Backend Handoff Artifacts** (in `backend/models/`):
  - `best_model.joblib` — the serialized RF_RAG classifier
  - `preprocessor.joblib` — the fitted ColumnTransformer (OneHot + StandardScaler)
  - `feature_names.json` — ordered feature names + metadata
  - `sample_test_patient.json` — end-to-end inference verification

### 3. Backend API & Live Inference
- **FastAPI Framework**: Exposes `POST /api/predict` and `POST /api/twin-patients` with strict Pydantic v2 schemas.
- **Live ML Inference** (`app/services.py`): When model artifacts are mounted, performs real RF_RAG inference. Gracefully falls back to a deterministic mock when artifacts are absent (CI environments).
- **Live RAG Retrieval**: Embeds incoming clinical notes, queries Qdrant for the top-3 most similar past encounters, and computes the semantic risk index.
- **Prescriptive LLM Care Plan**: When `LIVE_LLM_ENABLED=true`, calls Gemini 2.0 Flash to generate a structured 3-point discharge plan.

### 4. Clinician Authentication & Security
- **Firebase Auth Middleware** (`app/core/auth.py`): Validates Firebase JWT tokens on all protected routes.
- **Frontend Integration** (`src/contexts/AuthContext.tsx`): A `LoginModal` intercepts unauthenticated users. Firebase ID tokens are injected as Bearer headers in all API calls.
- **Dev Bypass**: When `FIREBASE_PROJECT_ID` is unset, the middleware is transparent (useful for CI/CD).

### 5. MLOps & Observability Stack
- **Docker Compose** orchestrates 6 services: FastAPI, Next.js, Qdrant, Redis, Prometheus, Grafana.
- **Prometheus** scrapes `/metrics` from the FastAPI backend.
- **Grafana** auto-provisions a dashboard (available at `http://localhost:3001`) visualising:
  - API request throughput (RPS)
  - P95/P99 inference latency
  - HTTP 2xx/4xx/5xx error rates

### 6. The Clinical Dashboard (Next.js 16)
- **Tech Stack**: Next.js 16, Tailwind CSS v4 (PostCSS pipeline), TypeScript.
- **Dark Mode UI**: High-contrast dark-mode mimicking Epic EHR aesthetics.
- **Dynamic Render**: Fetches from FastAPI to render the Risk Score Gauge, SDoH badges, Twin-Patient RAG cohort table, TreeSHAP force-plot, and GenAI care plan.

### 7. CI/CD Pipeline
- **GitHub Actions** (`deploy.yml`): Triggers on every push to `main`.
- **Quality Gates**: `ruff` linting, `bandit` security scan, `pytest` suite.
- **Deployment**: Builds and pushes the Docker image to GHCR under the `avert-care` organisation.

---

## 🚀 Universal Setup (For All Team Members)

> **One-command setup after `git pull develop`:**

```bash
# 1. Copy environment files
cp backend/.env.example backend/.env          # Fill in GEMINI_API_KEY
cp frontend/.env.example frontend/.env.production  # Fill in Firebase config

# 2. Spin up all 6 services
docker compose up -d --build

# 3. Seed Qdrant (first time only)
pip install -r scripts/requirements-seed.txt
python scripts/seed_rag_engine.py --csv data/raw/diabetic_data.csv

# 4. Open the dashboard
open http://localhost:3000
```

> **Service URLs after startup:**
> | Service | URL |
> |---|---|
> | Dashboard | http://localhost:3000 |
> | API Docs | http://localhost:8000/docs |
> | Qdrant UI | http://localhost:6333/dashboard |
> | Grafana | http://localhost:3001 (admin / avertcare) |
> | Prometheus | http://localhost:9090 |

---

## ⚙️ Feature Flags (in `backend/.env`)

| Flag | Default | Effect |
|---|---|---|
| `LIVE_RAG_ENABLED` | `true` | Uses live Qdrant retrieval; falls back to mock if `false` |
| `LIVE_LLM_ENABLED` | `false` | Calls Gemini 2.0 Flash; uses rule-based plan if `false` |

---

## 🎯 Next Steps

1. **Wire remaining frontend fields**: Expose more patient data fields in the UI (A1C result, medication type, etc.) so the RF_RAG model can use richer input data beyond the current defaults.
2. **Production deployment**: Configure Vercel for the frontend; deploy the FastAPI backend to Cloud Run or a VM with the Docker image from GHCR.
3. **CI/CD model update flow**: When M1 updates `.joblib` files, push to the `develop` branch — Docker restarts with the new model via the volume mount.
