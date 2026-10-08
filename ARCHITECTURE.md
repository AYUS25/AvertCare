# AvertCare — Project Architecture & File Reference

> **For new developers.** This document explains the purpose of every file and folder in the `AvertCare` repository, the system architecture, and the data flow from raw patient input to prescriptive clinical output.

---

## 🧭 System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CLINICIAN'S BROWSER                          │
│                    http://localhost:3000                              │
│                      Next.js 16 Dashboard                            │
│  Login → Firebase Auth JWT → Attach Bearer token to every API call  │
└─────────────────────┬────────────────────────────────────────────────┘
                      │ HTTPS / Bearer Token
                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND  :8000                            │
│  POST /api/predict        POST /api/twin-patients                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  auth.py → verify Firebase JWT                                  │ │
│  │  services.py → _get_ml_artifacts()                              │ │
│  │    ├── preprocessor.joblib (ColumnTransformer)                  │ │
│  │    ├── best_model.joblib   (RF_RAG classifier)                  │ │
│  │    └── feature_names.json                                       │ │
│  │  services.py → run_rag_retrieval()                              │ │
│  │    └── Qdrant :6333 → top-3 similar past patients               │ │
│  │  services.py → _generate_llm_care_plan()                        │ │
│  │    └── Gemini 2.0 Flash → 3-point discharge plan                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└────────────────────┬─────────────────────────────────────────────────┘
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌──────────────────────────┐
│  Qdrant :6333   │   │  Redis :6379              │
│  Vector DB      │   │  (Caching / Rate Limit)   │
│  clinical_cases │   └──────────────────────────┘
│  collection     │
└─────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│              PROMETHEUS :9090  →  GRAFANA :3001                      │
│  /metrics endpoint → time-series → ops dashboard                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Root Directory

| File / Folder | Purpose |
|---|---|
| `docker-compose.yml` | **Universal setup** — one file brings up all 6 services. Volume mounts ensure every developer shares the same ML artifacts and observability stack. |
| `WALKTHROUGH.md` | High-level project summary, feature flags, and setup instructions. |
| `ARCHITECTURE.md` | This file — detailed developer reference for every file and folder. |
| `FIREBASE_SETUP.md` | Step-by-step guide for manually initialising the Firebase project. |
| `README.md` | GitHub landing page — project overview and quick-start badge. |
| `.gitignore` | Excludes `.env`, `.venv/`, `__pycache__/`, `.pkl`/`.pt` large binaries. |
| `.github/workflows/deploy.yml` | GitHub Actions CI/CD: lint → test → build Docker image → push to GHCR. |

---

## 📁 `backend/` — FastAPI Inference Server

The FastAPI application that sits between the Next.js frontend and the ML models / Qdrant vector store.

### `backend/Dockerfile`
Multi-stage build:
1. **builder** stage: installs C-extension deps (shap, scikit-learn, numpy) with `gcc`.
2. **runner** stage: lean runtime image with only the compiled packages and app source.

### `backend/requirements.txt`
All Python runtime dependencies pinned to exact versions for reproducible builds across the team. Key packages:
- `fastapi`, `uvicorn` — API server
- `firebase-admin` — JWT verification
- `qdrant-client` — vector search
- `sentence-transformers` — clinical note embedding
- `scikit-learn`, `shap`, `pandas`, `numpy` — ML inference
- `google-genai` — Gemini LLM
- `prometheus-fastapi-instrumentator` — `/metrics` endpoint

### `backend/.env` / `backend/.env.example`
Runtime configuration. Never committed. `.env.example` is the template every team member copies and fills in. Key variables:
- `FIREBASE_PROJECT_ID` — enables JWT auth; blank = dev bypass mode
- `GEMINI_API_KEY` — enables Gemini care plan generation
- `LIVE_RAG_ENABLED` — switches from mock to live Qdrant retrieval
- `LIVE_LLM_ENABLED` — switches from rule-based to Gemini care plan

### `backend/app/main.py`
FastAPI application factory. Wires together:
- CORS middleware (allows `localhost:3000` and Vercel)
- Prometheus instrumentation (auto-exposes `/metrics`)
- API router
- `/health` and `/` probe endpoints

### `backend/app/schemas.py`
Pydantic v2 data models for all request/response types. Follows HL7 FHIR Encounter structure. Key types:
- `PatientEncounter` — inbound payload (age, diagnoses, clinical note, etc.)
- `PredictResponse` — risk score, SHAP features, SDoH flags, care plan
- `TwinPatientResponse` — RAG cohort with similarity scores

### `backend/app/services.py`
**The brain of the system.** Three responsibilities:
1. **`run_prediction()`** — loads ML artifacts, builds feature matrix, runs RF_RAG inference, computes SHAP values. Falls back to deterministic mock if model files absent.
2. **`run_rag_retrieval()`** — embeds clinical note, queries Qdrant, computes `rag_readmission_rate` and semantic neighbourhood risk index (SNRI).
3. **`_generate_llm_care_plan()`** — calls Gemini API with risk data and SHAP drivers; falls back to rule-based plan.

### `backend/app/core/auth.py`
FastAPI dependency (`require_auth`) that validates Firebase JWT ID tokens. In dev mode (no `FIREBASE_PROJECT_ID`), returns a dummy token object so the whole API is accessible without Firebase.

### `backend/app/core/config.py`
Pydantic `BaseSettings` class. Reads from environment / `.env` file. Single source of truth for all configuration values.

### `backend/app/api/routes.py`
Thin API router. Delegates all logic to `services.py`. Applies `require_auth` dependency to every endpoint.

### `backend/models/`
The mounted ML artifact directory. These files are committed to the repo so every developer immediately has the trained model after `git pull`:
- `best_model.joblib` — winning RF_RAG classifier (~15 MB)
- `preprocessor.joblib` — fitted ColumnTransformer
- `feature_names.json` — ordered feature list + metadata (147 features)
- `sample_test_patient.json` — reference inference test case

### `backend/tests/test_api.py`
pytest test suite with `TestClient`. Auth is mocked via `dependency_overrides`. Tests cover health probes, prediction, RAG retrieval, validation errors, and SDoH extraction.

---

## 📁 `frontend/` — Next.js Clinical Dashboard

### `frontend/Dockerfile`
Three-stage build (deps → builder → runner). Produces a minimal standalone Next.js image using `output: 'standalone'` in `next.config.ts`.

### `frontend/.env.production` / `frontend/.env.example`
Firebase Web SDK configuration and backend API URL. The `.production` file is used during the Docker build; `.example` is the template for new developers.

### `frontend/next.config.ts`
Minimal Next.js config. `output: 'standalone'` is critical — it enables Docker multi-stage build optimisation.

### `frontend/postcss.config.mjs`
Required for Tailwind CSS v4's PostCSS compilation pipeline to work correctly inside Docker's Turbopack build.

### `frontend/src/app/globals.css`
Tailwind v4 CSS using `@import "tailwindcss"`. Defines all CSS variables for the dark-mode design system (HSL color tokens for background, foreground, primary, border, etc.).

### `frontend/src/app/layout.tsx`
Root layout. Wraps the app in `AuthProvider` (Firebase) and `ThemeProvider`.

### `frontend/src/app/page.tsx`
Main dashboard page. Handles:
- Fetching patient data from FastAPI
- Rendering Risk Score Gauge, SDoH badges, Twin-Patient table, SHAP plot, care plan
- Injecting Firebase JWT into `Authorization: Bearer` header for every API call

### `frontend/src/contexts/AuthContext.tsx`
React context that wraps Firebase Auth. Provides `currentUser` and `getIdToken()` to the entire application.

### `frontend/src/components/LoginModal.tsx`
Full-screen modal that blocks dashboard access until the clinician signs in with Firebase Auth (email/password).

### `frontend/src/components/theme-provider.tsx`
`next-themes` wrapper for future light/dark mode toggle.

### `frontend/src/lib/firebase.ts`
Firebase app initialisation using `NEXT_PUBLIC_FIREBASE_*` environment variables.

---

## 📁 `ml_engine/` — Machine Learning Pipeline

All training, evaluation, and model serialisation code. Run by M1 (Data Scientist) on their local machine using Colab/Jupyter.

| File | Purpose |
|---|---|
| `run_phase1_2.py` | Data loading, cleaning, feature engineering, EDA |
| `split.py` | Stratified train/val/test split with `encounter_id` retention |
| `clean.py` | Data cleaning utilities (imputation, outlier removal, encoding) |
| `run_phase3_6.py` | Baseline model training (LR, RF) — Phases 3–6 without RAG |
| `run_phase4_6_rag.py` | **Main training script.** RAG ablation study across 6 models × 2 feature sets. Selects winner, generates SHAP plots, serialises artifacts to `backend/models/`. |
| `train_baselines.py` | Standalone baseline training helpers |
| `train_ft_transformer.py` | FT-Transformer training loop |
| `ft_transformer.py` | Custom PyTorch FT-Transformer classifier implementation |
| `evaluate.py` | AUROC, AUPRC, F1, Brier score computation + ROC/PR plot generation |
| `shap_explain.py` | `AvertCareExplainer` wrapper: computes global/local SHAP values for tree and neural net models |
| `predict.py` | Standalone batch inference utility |
| `rag_integration.py` | Helpers for joining the `rag_readmit_rate` feature onto train/val/test splits |
| `models/` | Serialised `.joblib` files for all 6 ablation models (committed to repo) |
| `reports/` | Ablation metrics CSV, markdown report, ROC/PR curve PNGs |
| `shap_output/` | SHAP summary bar + beeswarm plots, local explanation JSON |
| `metrics/` | Raw ablation metrics JSON |
| `data/processed/` | Intermediate processed splits with IDs |
| `requirements.txt` | Minimal ML-only dependencies (for local/Colab use) |

---

## 📁 `data/` — Dataset Storage

| Path | Purpose |
|---|---|
| `data/raw/diabetic_data.csv` | Original UCI Diabetes 130-US Hospitals dataset (10 years) |
| `data/processed/train_with_rag.csv` | M2's output — augmented training set with `rag_readmit_rate` column (~10K rows). Committed to repo so M1 can pull and train directly. |

---

## 📁 `scripts/` — Utility Scripts

| File | Purpose |
|---|---|
| `seed_rag_engine.py` | **One-time setup script.** Reads `diabetic_data.csv`, generates clinical notes, embeds them, and upserts into Qdrant. Also produces `train_with_rag.csv`. Run before enabling `LIVE_RAG_ENABLED=true`. |
| `requirements-seed.txt` | Dependencies for the seeder (qdrant-client, sentence-transformers, etc.) |

---

## 📁 `grafana/` — Observability

| Path | Purpose |
|---|---|
| `provisioning/prometheus.yml` | Prometheus scrape config — targets `backend:8000/metrics` |
| `provisioning/datasources/` | Grafana auto-datasource config (Prometheus URL) |
| `provisioning/dashboards/` | Auto-provisioned Grafana dashboard JSON (AvertCare IT Ops) |

---

## 📁 `notebooks/` — Jupyter Notebooks

Exploratory analysis notebooks used by M1 during the early phases of the project. Not used in production.

---

## 📁 `vector_db/` — Vector DB Init

Reserved for future Qdrant collection initialisation scripts. Currently a placeholder.

---

## 🔄 Data Flow (End-to-End)

```
CSV (diabetic_data.csv)
  → scripts/seed_rag_engine.py
      → generate clinical notes
      → embed with MiniLM-L6-v2
      → upsert into Qdrant (clinical_cases collection)
      → export data/processed/train_with_rag.csv

train_with_rag.csv
  → ml_engine/run_phase4_6_rag.py (M1)
      → RAG ablation (6 models)
      → Winner: RF_RAG
      → backend/models/{best_model, preprocessor, feature_names}.joblib|json

Docker Compose (docker compose up)
  → mounts backend/models → /app/models (read-only)
  → backend FastAPI loads artifacts on first request via @lru_cache

Clinician logs in via LoginModal (Firebase)
  → gets JWT ID Token
  → POST /api/predict with Bearer token
  → services.py:
      1. Validates JWT (auth.py)
      2. Builds patient DataFrame
      3. preprocessor.transform() → 146-dim feature vector
      4. best_model.predict_proba() → readmission probability
      5. shap.TreeExplainer() → top-5 SHAP drivers
      6. run_rag_retrieval() → top-3 twin patients from Qdrant
      7. Gemini 2.0 Flash → 3-point care plan (if enabled)
  → PredictResponse → Next.js dashboard renders results
```

---

## 👥 Team Roles

| Role | Responsibility |
|---|---|
| **M1 (Data Scientist)** | `ml_engine/` — data cleaning, model training, SHAP, serialisation |
| **M2 (You / Full-Stack)** | `backend/`, `frontend/`, `scripts/`, `docker-compose.yml`, CI/CD |

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| **RF_RAG over XGBoost** | M1's ablation showed RF+RAG achieves highest AUROC; XGBoost was out-of-scope for the task spec |
| **`backend/models/` committed to git** | Removes the "run training before anything works" problem — every developer has a working model after `git pull` |
| **Feature flag fallback** | `LIVE_RAG_ENABLED` / `LIVE_LLM_ENABLED` allow the app to run fully offline without Qdrant seeding or a Gemini API key |
| **Firebase Auth with dev bypass** | Blank `FIREBASE_PROJECT_ID` = no auth required — CI/CD passes without Firebase credentials |
| **Tailwind v4 PostCSS** | v4 requires `@import "tailwindcss"` and a `postcss.config.mjs` — the old v3 `@tailwind` directives crash Turbopack |
| **Volume mount for models** | M1 updates `.joblib` files → commit → team pulls → Docker restarts with new model, no image rebuild required |
