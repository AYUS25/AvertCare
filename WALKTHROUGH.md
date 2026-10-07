# AvertCare: Project Walkthrough & Roadmap

This document outlines everything we have built so far for the **Cognizant Hackathon (Use Case #6: Hospital Readmission Risk Prediction)**, our architectural decisions, and the exact roadmap for our next phases.

---

## 🏗️ What We Have Built So Far (The Scaffold)

We have successfully built a loosely coupled, production-ready full-stack scaffold. This means the system is fully wired from the frontend to the backend, using deterministic mock data. 

When our Data Scientist (M1) is ready, we simply swap out the mock logic for her trained models without breaking the system.

### 1. Multi-Container Orchestration (Docker)
- **`docker-compose.yml`**: Orchestrates four services on an isolated network:
  - **`backend`**: FastAPI application.
  - **`frontend`**: Next.js clinical dashboard.
  - **`qdrant`**: Vector Database for our Twin-Patient RAG retrieval.
  - **`redis`**: Caching and rate-limiting.
- A local volume `./ml_engine/models:/app/models` is configured so that when M1's `.pkl` files are dropped into the repository, the backend container reads them instantly without needing a rebuild.

### 2. The Backend API Gateway (FastAPI / Python)
- **Schema Contracts (`schemas.py`)**: Built strictly with Pydantic v2. The payload fields (`patient_id`, `age`, `clinical_note`, etc.) are mapped to HL7 FHIR standards.
- **Thin Routing (`routes.py`)**: Exposes two endpoints:
  - `POST /api/predict`: Returns readmission risk, SHAP explainability factors, SDoH flags, and a GenAI Care Plan.
  - `POST /api/twin-patients`: Simulates the RAG engine returning historical patient twins and the calculated semantic risk index.
- **Mock Service Layer (`services.py`)**: All business logic is isolated here. We have deterministic mock logic that dynamically updates based on the JSON payload. This is the **exact swap point** for real model inference later.
- **Testing (`tests/test_api.py`)**: 100% passing Pytest suite validating our data contracts and SDoH keyword extraction logic.
- **Observability (`main.py`)**: Prometheus instrumentator is already attached, exposing the `/metrics` endpoint for MLOps tracking.

### 3. The Clinical Dashboard (Next.js / TypeScript)
- **Tech Stack**: Next.js 15 (App Router), Tailwind CSS, and `lucide-react`.
- **Dark Mode Aesthetic**: Implemented `next-themes` and a `ThemeProvider` to strictly enforce a high-contrast dark-mode UI, reducing alert fatigue and mimicking modern Epic EHR interfaces.
- **Zero-Bloat UI (`page.tsx`)**: Instead of relying heavily on interactive CLI installers which can cause bloat, the UI is built using raw Tailwind classes that visually match Shadcn/UI standards. 
- **Dynamic Fetching**: The dashboard holds a mock FHIR payload state. Clicking "Analyze Patient Record" asynchronously fires requests to our local FastAPI endpoints and gracefully renders:
  - The Risk Score Gauge.
  - Extracted Social Determinants of Health (SDoH) warning badges.
  - A table of Twin-Patient RAG cohorts.
  - A visual CSS-based TreeSHAP force-plot breakdown.
  - A formatted GenAI prescriptive care plan with CMS Penalty ROI estimates.

### 4. CI/CD & DevOps
- **Dockerfiles**: We created highly optimized Dockerfiles for both services (e.g., using `node:20-alpine` standalone for the frontend and `python:3.11-slim` for the backend).
- **GitHub Actions (`deploy.yml`)**: An automated 4-stage pipeline is live. On every push to `main`, it:
  1. Lints (Ruff) and runs Security Scans (Bandit).
  2. Runs the Pytest suite.
  3. Builds and pushes the Backend Docker image to GHCR.
  4. Deploys the Frontend directly to Vercel.

---

## 🚀 What We Are Building Next (The Roadmap)

Now that the 0-to-1 scaffolding is complete, here is my understanding of what we must tackle next to complete the hackathon objective.

### Phase 1: Model Injection (Handoff from M1)
1. **Load the Models**: Drop M1's trained XGBoost `.pkl` and Bio_ClinicalBERT artifacts into `ml_engine/models/`.
2. **Wire Inference**: Open `backend/app/services.py`, delete the mock logic in `run_prediction`, and load the real models using `joblib`/`xgboost`.
3. **Wire Qdrant**: Replace the mock twin-patient logic in `run_rag_retrieval`. We will embed the inbound `clinical_note` using our BERT model, query the live Qdrant container, and fetch the real $RAG_{risk}$ index.

### Phase 2: Live LLM Care Plan Generation
- Currently, the 3-point discharge plan is generated via simple `if/else` logic.
- We will integrate the Gemini API (or OpenAI API) into the backend. The LLM prompt will synthesize the XGBoost risk score, the SHAP features, and the successful interventions from the RAG twins to generate a highly personalized, natural-language discharge plan.

### Phase 3: Firebase Auth Integration (Clinician Login)
- Hospital systems require secure access. We will add a login page to the Next.js frontend using Firebase Auth.
- We will pass the Firebase JWT token in the `Authorization` header of our API calls.
- In FastAPI, we will add a dependency to decode and validate the JWT before processing the FHIR payload.

### Phase 4: MLOps Dashboard (Grafana)
- Since the `/metrics` endpoint is already live on FastAPI, we will add a Grafana container to `docker-compose.yml`.
- We will build a dashboard for IT Ops to monitor inference latency (ms), ensuring we meet the "Real-Time Decision" SLAs outlined in the PDD.

---
*The system is highly decoupled. M1 can focus strictly on model weights, while I handle the integration and frontend state.*
