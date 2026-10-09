# Data Dictionary
## Dataset
Source: UCI ML Repository — Diabetes 130-US Hospitals (1999–2008), id=296.  
Original: ~101,766 encounters × 50 features.  
After cleaning: see row counts below.

## Target
| Original value | `readmitted_binary` |
|---|---|
| `<30` | **1** (positive: readmitted within 30 days) |
| `>30` | 0 |
| `NO`  | 0 |

## Cleaning Steps (in order)
1. `readmitted` → `readmitted_binary` (see Target above); original column dropped.
2. All `?` values replaced with NaN.
3. **`weight`**: dropped — >96% missing; not recoverable.
4. **`discharge_disposition_id`** rows 11, 13, 14, 19, 20, 21 removed (death / hospice).
5. **`gender == 'Unknown/Invalid'`** rows removed (< 5 rows).
6. **`race`**: NaN → `'Unknown'` (2.2% of rows had missing race).
7. **`payer_code`**: NaN → `'Unknown'` (39.7% missing — information-free but useful category).
8. **`medical_specialty`**: NaN → `'Unknown'` (48.9% missing). Specialties representing
   < 1% of encounters collapsed into `'Other'`.
9. **`max_glu_serum`**: NaN → `'Not_Tested'` (94.8% not performed — test not ordered ≠ normal).
   Sentinel `'Not_Tested'` is used instead of `'None'` because pandas.read_csv() treats
   the string `'None'` as NaN by default, making it unsafe as a CSV sentinel.
10. **`A1Cresult`**: NaN → `'Not_Tested'` (83.1% not performed — same rationale).
11. **`admission_type_id`**, **`discharge_disposition_id`**, **`admission_source_id`**:
    converted to string categories; values < 0.5% of encounters collapsed to `'Other'`.
12. Constant/near-constant columns dropped automatically.
13. **`diag_1`**, **`diag_2`**, **`diag_3`**: each ICD-9 code grouped into one of:
    `Circulatory`, `Respiratory`, `Digestive`, `Genitourinary`, `Neoplasms`,
    `Musculoskeletal`, `Injury`, `Diabetes`, `Other`. Original columns dropped.
14. **`age`**: band string (e.g. `[60-70)`) → numeric midpoint (`age_numeric = 65`).
15. Medication columns present as `No / Steady / Up / Down`. Columns where > 99% of
    encounters are `'No'` are dropped (near-constant). Retained: see CSV columns.

## Engineered Features
| Feature | Formula |
|---|---|
| `total_prior_visits` | `number_outpatient + number_emergency + number_inpatient` |
| `num_med_changes` | count of medication columns with value `'Up'` or `'Down'` |
| `num_meds_active` | count of medication columns with value ≠ `'No'` |
| `age_numeric` | numeric midpoint of the original 10-year age band |

## ID Columns (MUST NOT be model features)
| Column | Purpose |
|---|---|
| `encounter_id` | Unique encounter identifier — retained for M2/RAG matching only |
| `patient_nbr` | Patient identifier — used to enforce patient-level split grouping |

## Split Strategy
- 80 / 10 / 10 (train / val / test)
- Stratified by `readmitted_binary` (preserves class ratio in each split)
- Grouped by `patient_nbr`: ALL encounters of one patient go to exactly ONE split
- No patient overlap between any pair of splits (verified with assertions)
- Random seed: 42 (fixed throughout the project)
- Algorithm: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`

## Class Imbalance
- Positive class (`readmitted_binary=1`) ≈ 11.4% across all splits
- Strategy: `class_weight='balanced'` applied to training estimators only
- NO resampling (SMOTE etc.) is applied to validation or test data
- If SMOTE is used in Phase 3, it is applied ONLY inside the training pipeline
  after the train/val/test split, never before

## Deferred Preprocessing (Phase 3 pipeline)
The `*_with_ids.csv` files retain HUMAN-READABLE values (string categories,
unscaled numerics) so M2 can use them directly for RAG.
The following transformations are applied in the Phase-3 sklearn pipeline:
- Numeric features: median imputation → StandardScaler
- Categorical features: mode imputation → OneHotEncoder (LR/RF)
  OR OrdinalEncoder (FT-Transformer)
Fitted preprocessors are saved as:
- `ml_engine/data/processed/preprocessor_onehot.joblib`
- `ml_engine/data/processed/preprocessor_ft.joblib`

## Numeric Features
  age_diag_interaction, age_inpatient_interaction, age_numeric, clinical_complexity, diagnoses_per_day, emergency_ratio, emergency_sq, has_prior_emergency, has_prior_inpatient, has_prior_outpatient, high_utilizer, inpatient_ratio, inpatient_sq, lab_med_ratio, labs_per_day, med_change_ratio, meds_per_day, num_lab_procedures, num_med_changes, num_medications, num_meds_active, num_procedures, number_diagnoses, number_emergency, number_inpatient, number_outpatient, polypharmacy, procedures_per_day, time_in_hospital, total_prior_visits

## Categorical Features
  A1Cresult, admission_source_id, admission_type_id, change, diabetesMed, diag_1_group, diag_2_group, diag_3_group, discharge_disposition_id, gender, glimepiride, glipizide, glyburide, insulin, max_glu_serum, medical_specialty, metformin, payer_code, pioglitazone, race, repaglinide, rosiglitazone
