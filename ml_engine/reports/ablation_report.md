# AvertCare M1 — Phase 4 RAG Ablation Report

## Ablation Matrix (Test Set Metrics)

| Model | Configuration | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Test Specificity | AUROC Δ (RAG) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Tabular Only | 0.668 | 0.2331 | 0.2765 | 0.1915 | 0.4973 | 0.73 | nan |
| Logistic Regression | Tabular + RAG | 0.6918 | 0.3135 | 0.3167 | 0.3041 | 0.3304 | 0.9028 | 0.0238 |
| Random Forest | Tabular Only | 0.6725 | 0.2324 | 0.2801 | 0.2038 | 0.4479 | 0.775 | nan |
| Random Forest | Tabular + RAG | 0.6971 | 0.3115 | 0.3119 | 0.2668 | 0.3754 | 0.8673 | 0.0246 |
| FT-Transformer | Tabular Only | 0.6607 | 0.2185 | 0.2619 | 0.1679 | 0.5945 | 0.6212 | nan |
| FT-Transformer | Tabular + RAG | 0.6662 | 0.2679 | 0.286 | 0.2271 | 0.386 | 0.8311 | 0.0055 |

_AUROC Δ = RAG model AUROC minus Tabular-only AUROC (positive = improvement)_

## Winner Selection Criteria

Winner selected by: **highest Test AUROC** (primary), then **Test F1 at optimal threshold** (tiebreak).

Note: XGBoost was NOT part of the M1 task specification. The ablation covers LR, RF, and FT-Transformer as required. If the backend contract demands `xgboost_model.pkl`, this represents a naming mismatch that must be reconciled with the team lead before renaming the actual winning model.
