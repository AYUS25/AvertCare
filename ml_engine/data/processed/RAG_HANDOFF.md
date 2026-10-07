# RAG Feature Handoff
M2, please provide the RAG feature as one CSV per split at:
- `ml_engine/data/rag/rag_features_train.csv`
- `ml_engine/data/rag/rag_features_val.csv`
- `ml_engine/data/rag/rag_features_test.csv`

Columns required: `encounter_id`, `rag_neighborhood_readmit_rate`
The retrieval index must be built from TRAIN encounters ONLY.
CRITICAL: For TRAIN rows, retrieval MUST EXCLUDE the row itself AND all other encounters of the same `patient_nbr` (leave-one-out) to prevent leakage.
