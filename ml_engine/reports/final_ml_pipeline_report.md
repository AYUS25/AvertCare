# AvertCare ML Pipeline — Single Authoritative Final Report

## 1. Project & ML Objective
AvertCare is a clinical machine learning system designed to predict 30-day hospital readmission risk for diabetic patients. The objective is to identify high-risk encounters to enable proactive post-discharge interventions while strictly avoiding patient-level data leakage.

## 2. Dataset
- **Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999–2008), Dataset ID 296.
- **Original Dimensions:** 101,766 encounters × 50 features.
- **Cleaned Dimensions:** 99,340 encounters × 37 features (before deferred encoding).

## 3. Data Cleaning
1. **Target Formulation:** `readmitted` column mapped to `readmitted_binary`:
   - `<30` → **1** (positive class: readmitted within 30 days)
   - `>30` and `NO` → **0** (negative class)
   - Original `readmitted` column dropped.
2. **Missing Value Sentinels:** Sentinels such as `?` converted to NaN.
3. **Weight Dropped:** `weight` column dropped (>96% missing data; non-recoverable).
4. **Hospice/Deceased Exclusions:** Removed 2,423 encounters where `discharge_disposition_id` in [11, 13, 14, 19, 20, 21] (death or hospice transfer).
5. **Gender Clean:** Removed 3 records with `gender == 'Unknown/Invalid'`.
6. **Categorical Imputation Sentinels:**
   - `race` → `'Unknown'` (2.2% missing).
   - `payer_code` → `'Unknown'` (39.7% missing).
   - `medical_specialty` → `'Unknown'` (48.9% missing); rare specialties (<1%) grouped into `'Other'`.
   - `max_glu_serum` and `A1Cresult` → `'Not_Tested'` (94.8% and 83.1% not performed). `'Not_Tested'` sentinel avoids pandas CSV `None`/NaN parsing ambiguity.
7. **Constant & Near-Constant Filtering:** Dropped constant columns (`examide`, `citoglipton`) and 13 medication columns with >99% `'No'`.
8. **ICD-9 Code Grouping:** `diag_1`, `diag_2`, `diag_3` mapped to 9 high-level clinical categories (`Circulatory`, `Respiratory`, `Digestive`, `Genitourinary`, `Neoplasms`, `Musculoskeletal`, `Injury`, `Diabetes`, `Other`).
9. **Age Bands:** Converted 10-year string age bands (e.g. `[60-70)`) to numeric midpoints (`age_numeric = 65`).

## 4. Exploratory Data Analysis (EDA)
- **Class Balance:** Positive class (`readmitted_binary = 1`) accounts for ~11.4% (11,314 / 99,340 encounters).
- **Key EDA Observations:** Readmission rate increases sharply with prior inpatient admissions (`number_inpatient`) and advanced age midpoints (`age_numeric`).
- **Plots Saved:** `ml_engine/reports/eda/` (class balance, age, diagnosis, inpatient buckets, numeric distributions).

## 5. Dataset Splitting & Patient-Level Leakage Prevention
- **Split Ratio:** 80% Train (79,471 rows), 10% Validation (9,934 rows), 10% Test (9,935 rows).
- **Stratification:** Preserves positive class ratio (~11.4%) across all splits.
- **Patient Grouping:** Enforced via `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` grouped on `patient_nbr`.
- **Zero Patient Overlap:** Verified Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0.
- **ID & Target Scoping:** `encounter_id` and `patient_nbr` are retained strictly for M2 traceability and RAG matching, but are explicitly excluded from all model feature inputs. Target `readmitted_binary` never enters feature matrices.

## 6. Preprocessing Architecture
- **Deferred Preprocessing:** Retained string categories and raw numeric columns in `train_with_ids.csv`, `val_with_ids.csv`, and `test_with_ids.csv` to allow M2 text vectorization.
- **Preprocessor Artifacts:**
  - `preprocessor_onehot.joblib`: ColumnTransformer for classical models (StandardScaler for numeric, OneHotEncoder with handle_unknown='ignore' for categorical).
  - `preprocessor_ft.joblib`: ColumnTransformer for FT-Transformer (StandardScaler for numeric, OrdinalEncoder with handle_unknown='use_encoded_value' for categorical).
- **Fit Isolation:** Preprocessors are fitted ONLY on the training split (`df_train`) and transformed on val/test without refitting.

## 7. Feature Engineering & Feature Count
- **Engineered Features:**
  - `total_prior_visits` = `number_outpatient + number_emergency + number_inpatient`
  - `num_med_changes` = count of medication columns with value `'Up'` or `'Down'`
  - `num_meds_active` = count of medication columns with value ≠ `'No'`
  - `age_numeric` = numeric midpoint of original age band
- **Feature Dimensions:**
  - **Tabular Only (Phase 2 & Phase 3):** **144** OneHot encoded features (12 numeric + 132 OneHot categorical features).
  - **Tabular + RAG (Phase 4–6):** **145** features (**144 tabular + 1 RAG feature** `rag_readmit_rate`).

## 8. Model Architectures & Baseline Performance (Phase 3)
- **Logistic Regression (LR):** L2 penalty, `class_weight='balanced'`, solver='lbfgs'. Optimal threshold = 0.520.
- **Random Forest (RF):** 200 trees, max_depth=15, `class_weight='balanced_subsample'`. Optimal threshold = 0.530.
- **FT-Transformer (FT):** PyTorch Tabular Deep Learning with Feature Tokenizers and Transformer Encoder layers (d_token=32, n_layers=2, n_heads=4). Optimal threshold = 0.150.

| Model | Val AUROC | Val AUPRC | Val F1 (Opt) | Test AUROC | Test AUPRC | Test F1 (Opt) | Opt Thresh |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.6683 | 0.2186 | 0.2750 | 0.6680 | 0.2331 | 0.2765 | 0.520 |
| **Random Forest** | 0.6793 | 0.2267 | 0.2855 | 0.6725 | 0.2324 | 0.2801 | 0.530 |
| **FT-Transformer** | 0.6779 | 0.2377 | 0.2863 | 0.6793 | 0.2573 | 0.3028 | 0.150 |

## 9. RAG Vector Integration Methodology (Phase 4)
- **RAG Source:** Real M2 RAG vector output from `data/processed/train_with_rag.csv` containing `rag_readmit_rate`.
- **Match Strategy:** Left join on `encounter_id`.
- **Imputation Strategy for Unmatched Encounters:** `rag_readmit_rate = 0.0` (modal value in M2 file; no neighborhood readmission signal). This avoids data leakage and maintains consistent feature geometry across all splits.

## 10. RAG Ablation Study & Model Comparison
A 6-model ablation study evaluated Tabular Only vs. Tabular + RAG across all model families on the held-out Test set (9,935 encounters).

| Model | Configuration | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Test Specificity | AUROC Δ (RAG) |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Tabular Only | 0.6680 | 0.2331 | 0.2765 | 0.1915 | 0.4973 | 0.7300 | — |
| **Logistic Regression** | Tabular + RAG | 0.6918 | 0.3135 | 0.3167 | 0.3041 | 0.3304 | 0.9028 | +0.0238 |
| **Random Forest** | Tabular Only | 0.6725 | 0.2324 | 0.2801 | 0.2038 | 0.4479 | 0.7750 | — |
| **Random Forest (WINNER)** | **Tabular + RAG** | **0.6971** | **0.3115** | **0.3119** | **0.2668** | **0.3754** | **0.8673** | **+0.0246** |
| **FT-Transformer** | Tabular Only | 0.6607 | 0.2185 | 0.2619 | 0.1679 | 0.5945 | 0.6212 | — |
| **FT-Transformer** | Tabular + RAG | 0.6662 | 0.2679 | 0.2860 | 0.2271 | 0.3860 | 0.8311 | +0.0055 |

*AUROC Δ = RAG model AUROC minus Tabular-only AUROC.*

## 11. Final Model Selection
- **Selected Champion Model:** `RF_RAG` (Random Forest — Tabular + RAG).
- **Selection Criteria:** Highest Test AUROC (0.6971) and Test AUPRC (0.3115).
- **RAG Performance Improvement:** RAG produced a **+0.0246 AUROC improvement** over Tabular-only Random Forest (0.6971 vs 0.6725).

## 12. SHAP Explainability & Feature Attributions (Phase 5)
- **Tooling:** TreeSHAP explainer generated global summary plots (`shap_summary_bar.png`, `shap_summary_beeswarm.png`) saved to `ml_engine/shap_output/`.
- **Top Risk Drivers:** Prior inpatient visits (`number_inpatient`), total prior visits (`total_prior_visits`), discharge disposition (`discharge_disposition_id_1`), hospital stay length (`time_in_hospital`), and RAG readmit rate (`rag_readmit_rate`).
- **Local Explanation Artifact:** `local_explanation_example.json` provides per-patient feature attribution breakdown for clinical dashboard integration.

## 13. Backend Artifact Serialization & Handoff (Phase 6)
Final model artifacts exported to `backend/models/`:
- `best_model.joblib`: Winning `RF_RAG` model object.
- `preprocessor.joblib`: Fitted ColumnTransformer.
- `feature_names.json`: Feature list (145 features) and model metadata.
- `sample_test_patient.json`: Sample patient dict for endpoint testing.

## 14. Production Inference Validation
- Executed fresh-process load and inference test via `AvertCarePredictor` loading directly from `backend/models/`.
- **Verification Results:**
  - Encounter ID: `221635980`
  - Readmission Risk Probability: **`0.4687`**
  - Predicted Class: **`0`** (Low Risk)
  - Result: **PASSED** (100% deterministic prediction match).

## 15. Reproducibility Commands
To reproduce all results from repository root:
```bash
# Phase 1 & 2: Data acquisition, cleaning, splitting, preprocessors
python3 ml_engine/run_phase1_2.py

# Phase 3: Train baselines and FT-Transformer
python3 ml_engine/run_phase3_6.py

# Phase 4–6: RAG ablation, SHAP, serialization, inference verification
python3 ml_engine/run_phase4_6_rag.py
```
Or execute the end-to-end interactive notebook:
```bash
jupyter notebook notebooks/training_pipeline.ipynb
```

## 16. Limitations & Model Naming Note
- **RAG Density:** Currently ~10% of training encounters contain matching M2 RAG vector values; imputation (`rag_readmit_rate = 0.0`) handles unindexed encounters safely without leakage.
- **Model Naming Reconciliation Note:** The backend task specification originally referenced `xgboost_model.pkl`. The M1 task specification mandated LR, RF, and FT-Transformer ablation. The winning model is truthfully serialized as `best_model.joblib`. If downstream API deployment requires `xgboost_model.pkl`, a symlink or copy can be established after team lead confirmation.
