# RAG Feature Handoff — M1 → M2 Interface Contract

## Files to Provide (one CSV per split)

| File | Description |
|------|-------------|
| `ml_engine/data/rag/rag_features_train.csv` | RAG feature for training encounters |
| `ml_engine/data/rag/rag_features_val.csv`   | RAG feature for validation encounters |
| `ml_engine/data/rag/rag_features_test.csv`  | RAG feature for test encounters |

## Required Columns

Each CSV must contain exactly:

```
encounter_id, rag_neighborhood_readmit_rate
```

- `encounter_id` — matches encounter_id in the corresponding `*_with_ids.csv`.
- `rag_neighborhood_readmit_rate` — float in [0, 1]; mean readmission rate of
  the k nearest neighbour encounters retrieved from the **TRAIN** index.

## CRITICAL Leakage Rules (MUST be enforced programmatically)

### Rule 1 — Index built from TRAIN encounters ONLY

The vector retrieval index must be constructed from training encounters only.

DO NOT build a global index containing train + validation + test rows.

### Rule 2 — Leave-one-out for TRAIN rows

When computing `rag_neighborhood_readmit_rate` for a TRAIN encounter, the
retrieval MUST EXCLUDE:

1. The current encounter itself.
2. ALL other encounters belonging to the same `patient_nbr`.

This prevents a patient's own history from leaking into their own RAG feature.

### Rule 3 — Validation and Test rows

For VAL and TEST encounters, retrieve from the full TRAIN index (no exclusions
needed because these patients are entirely absent from the train set by design).

## How M1 uses the RAG feature

In Phase 4, M1 will:
1. Left-join the RAG CSVs onto the `*_with_ids.csv` on `encounter_id`.
2. Add `rag_neighborhood_readmit_rate` as an extra numeric input feature.
3. Train and evaluate all three models with and without this feature (ablation).

If the RAG CSVs are missing at Phase 4 time, the ablation will be run
without the RAG feature and a warning will be printed.
