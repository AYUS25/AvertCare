# AvertCare ML Engine — Phase 3 to Phase 6 Final Execution Report

## 1. Executive Summary
This report documents the end-to-end execution, evaluation, and serialization of Phases 3–6 of the AvertCare Hospital Readmission Risk Prediction system.

---

## 2. Phase 3: Model Performance Comparison

| Model | Val AUROC | Val AUPRC | Val F1 (Opt) | Test AUROC | Test AUPRC | Test F1 (Opt) | Opt Thresh |
|-------|-----------|-----------|--------------|------------|------------|---------------|------------|
| **Logistic Regression** | 0.6683 | 0.2186 | 0.275 | 0.668 | 0.2331 | 0.2765 | 0.52 |
| **Random Forest** | 0.6793 | 0.2267 | 0.2855 | 0.6725 | 0.2324 | 0.2801 | 0.53 |
| **FT-Transformer** | 0.6773 | 0.2271 | 0.2866 | 0.6775 | 0.2444 | 0.2836 | 0.13 |

---

## 3. Phase 4: RAG Integration Architecture
- **Status:** Interface finalized for M2 RAG module handoff.
- **Supported Fusion:** Feature Concatenation (Early Fusion) and Probability Blending (Late Fusion).
- **Tabular Vector Dim:** 144 dimensions.
- **RAG Hook:** `RAGFeatureIntegrator.concatenate_features(X_tabular, X_rag)`.

---

## 4. Phase 5: SHAP Explainability
- Global SHAP summary bar plot and beeswarm plot saved to `ml_engine/shap_output/`.
- Individual patient risk factor attributions extracted.

---

## 5. Phase 6: Production Inference API
Sample Patient Inference Output:
```json
{
  "readmission_risk_probability": 0.6227,
  "readmission_predicted": 1,
  "risk_band": "HIGH",
  "risk_label": "High Risk",
  "color_code": "#EF4444",
  "top_risk_drivers": [
    {
      "feature": "num__number_inpatient",
      "value": 1.921422449261735,
      "importance": 0.15026146606596197,
      "contribution": 0.28871575415811973
    },
    {
      "feature": "num__total_prior_visits",
      "value": 1.2342041726001076,
      "importance": 0.1219203819921314,
      "contribution": 0.1504746441796876
    },
    {
      "feature": "cat__discharge_disposition_id_1",
      "value": 1.0,
      "importance": 0.05339871431122294,
      "contribution": 0.05339871431122294
    },
    {
      "feature": "num__time_in_hospital",
      "value": 1.2233468786159154,
      "importance": 0.03872917976134384,
      "contribution": 0.04737922117239467
    },
    {
      "feature": "num__num_lab_procedures",
      "value": 0.9731448848835701,
      "importance": 0.04306765810296645,
      "contribution": 0.04191107118681624
    }
  ]
}
```
