# AvertCare M1 — Phase 4–6 Final Execution Report

## Phase 4: RAG Ablation

**RAG feature:** `rag_readmit_rate` (from M2 `train_with_rag.csv`)

**Imputation strategy for unmatched rows:** `rag_readmit_rate = 0.0` (modal value in M2 file; no data leakage)

| Model | AUROC | AUPRC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression — Tabular Only | 0.668 | 0.2331 | 0.2765 | 0.1915 | 0.4973 |
| Logistic Regression — Tabular + RAG | 0.6918 | 0.3135 | 0.3167 | 0.3041 | 0.3304 |
| Random Forest — Tabular Only | 0.6725 | 0.2324 | 0.2801 | 0.2038 | 0.4479 |
| Random Forest — Tabular + RAG **← WINNER** | 0.6971 | 0.3115 | 0.3119 | 0.2668 | 0.3754 |
| FT-Transformer — Tabular Only | 0.6542 | 0.2195 | 0.2733 | 0.1899 | 0.4876 |
| FT-Transformer — Tabular + RAG | 0.6634 | 0.2632 | 0.2878 | 0.212 | 0.4479 |

**Winner:** `RF_RAG` — Random Forest — Tabular + RAG

**RAG AUROC Impact:** RAG produced a +0.0246 AUROC improvement over Tabular Only (0.6971 vs 0.6725).

**Feature Count:** 145 total features (144 OneHot tabular features + 1 RAG feature `rag_readmit_rate`).

**XGBoost naming note:** The M1 task spec requires ablation across LR/RF/FT-Transformer. XGBoost was not in scope. The winning model is serialized under a truthful filename. Team lead must confirm filename reconciliation before renaming to `xgboost_model.pkl`.

## Phase 5: SHAP

- Global SHAP bar + beeswarm plots: `ml_engine/shap_output/`
- Local explanation example: `ml_engine/shap_output/local_explanation_example.json`
- `rag_readmit_rate` included in features: **True**

## Phase 6: Serialization & Backend Handoff

| Artifact | Path |
|---|---|
| Final model | `backend/models/best_model.joblib` |
| Preprocessor | `backend/models/preprocessor.joblib` |
| Feature names | `backend/models/feature_names.json` |
| Sample patient | `backend/models/sample_test_patient.json` |

## Inference Test

- Result: **PASSED**
- Probability from fresh load: `0.4687`
- Predicted class: `0`
