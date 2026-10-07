"""
Master Pipeline Orchestration Script for Phase 3 - Phase 6
AvertCare Hospital Readmission Prediction System
"""

import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluate import compute_metrics, find_optimal_threshold, plot_roc_curves, plot_pr_curves, save_metrics_summary
from train_baselines import train_and_evaluate_baselines
from train_ft_transformer import train_and_evaluate_ft_transformer
from rag_integration import RAGFeatureIntegrator
from shap_explain import AvertCareExplainer
from predict import AvertCarePredictor, predict_patient_risk


def run_full_phase3_6_pipeline():
    print("=" * 80)
    print("AVERTCARE ML ENGINE: PHASE 3 TO PHASE 6 MASTER EXECUTION")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. PHASE 3: MODEL TRAINING & EVALUATION
    # ---------------------------------------------------------
    print("\n[STEP 1/5] Phase 3: Training Baseline Models (Logistic Regression & Random Forest)...")
    baseline_metrics, baseline_test_probs = train_and_evaluate_baselines()

    print("\n[STEP 2/5] Phase 3: Training Deep Learning Model (FT-Transformer)...")
    ft_metrics, ft_test_probs = train_and_evaluate_ft_transformer()

    # Combine all test probabilities for combined comparison plot
    data_dir = "ml_engine/data/processed"
    df_test = pd.read_csv(os.path.join(data_dir, "test_with_ids.csv"))
    y_test = df_test["readmitted_binary"].values

    all_test_probs = {
        "Logistic Regression": baseline_test_probs["Logistic Regression"],
        "Random Forest": baseline_test_probs["Random Forest"],
        "FT-Transformer": ft_test_probs,
    }

    reports_dir = "ml_engine/reports/model_evaluation"
    os.makedirs(reports_dir, exist_ok=True)

    plot_roc_curves(all_test_probs, y_test, os.path.join(reports_dir, "master_roc_curves.png"))
    plot_pr_curves(all_test_probs, y_test, os.path.join(reports_dir, "master_pr_curves.png"))
    print(f"Master comparison plots generated -> {reports_dir}/")

    # Combine metrics
    master_metrics = {
        "Logistic_Regression": baseline_metrics["Logistic_Regression"],
        "Random_Forest": baseline_metrics["Random_Forest"],
        "FT_Transformer": ft_metrics["FT_Transformer"],
    }
    save_metrics_summary(master_metrics, "ml_engine/metrics/master_model_comparison.json")

    # ---------------------------------------------------------
    # 2. PHASE 4: RAG INTEGRATION INTERFACE VERIFICATION
    # ---------------------------------------------------------
    print("\n[STEP 3/5] Phase 4: Verifying RAG Integration Architecture...")
    rag_integrator = RAGFeatureIntegrator(
        tabular_preprocessor_path="ml_engine/data/processed/preprocessor_onehot.joblib",
        model_path="ml_engine/models/random_forest.joblib",
    )
    rag_spec = rag_integrator.get_integration_spec()
    print("RAG Handoff Specification:")
    print(json.dumps(rag_spec, indent=2))

    # Test feature concatenation with mock RAG dimension (verification only, not saved to disk)
    X_tab_dummy = np.random.randn(5, 144)
    X_rag_dummy = np.random.randn(5, 768)  # e.g., 768-dim BERT embeddings
    X_comb_dummy = rag_integrator.concatenate_features(X_tab_dummy, X_rag_dummy)
    assert X_comb_dummy.shape == (5, 144 + 768), f"RAG concat failed, got shape {X_comb_dummy.shape}"
    print(f"RAG feature concatenation verified -> Output shape: {X_comb_dummy.shape} ✓")

    # ---------------------------------------------------------
    # 3. PHASE 5: SHAP EXPLAINABILITY & RISK ATTRIBUTION
    # ---------------------------------------------------------
    print("\n[STEP 4/5] Phase 5: Generating SHAP Explainability Plots...")
    rf_model = joblib.load("ml_engine/models/random_forest.joblib")
    prep_obj = joblib.load("ml_engine/data/processed/preprocessor_onehot.joblib")
    preprocessor = prep_obj["preprocessor"] if isinstance(prep_obj, dict) else prep_obj
    feature_names = list(preprocessor.get_feature_names_out()) if hasattr(preprocessor, "get_feature_names_out") else None

    df_train = pd.read_csv(os.path.join(data_dir, "train_with_ids.csv"))
    # Use sample of 500 records for fast SHAP calculation
    df_sample = df_train.sample(n=500, random_state=42)
    X_sample_trans = preprocessor.transform(df_sample)

    explainer = AvertCareExplainer(rf_model, feature_names=feature_names)
    shap_plots = explainer.generate_summary_plots(X_sample_trans, output_dir="ml_engine/shap_output")
    print(f"SHAP plots generated: {shap_plots} ✓")

    # Explain single patient
    single_explanation = explainer.explain_patient_instance(X_sample_trans[0:1], top_k=3)
    print("Sample Patient SHAP Attribution (Top 3 Drivers):")
    print(json.dumps(single_explanation["top_risk_drivers"], indent=2))

    # ---------------------------------------------------------
    # 4. PHASE 6: END-TO-END INFERENCE PIPELINE VERIFICATION
    # ---------------------------------------------------------
    print("\n[STEP 5/5] Phase 6: Verifying End-to-End Inference Pipeline...")
    predictor = AvertCarePredictor(
        preprocessor_path="ml_engine/data/processed/preprocessor_onehot.joblib",
        model_path="ml_engine/models/random_forest.joblib",
    )

    # Sample raw patient record from raw dataset format
    raw_sample_patient = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "time_in_hospital": 8,
        "num_lab_procedures": 62,
        "num_procedures": 2,
        "num_medications": 21,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 3,
        "number_diagnoses": 9,
        "max_glu_serum": "None",
        "A1Cresult": ">8",
        "metformin": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": "Steady",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "examide": "No",
        "citoglipton": "No",
        "insulin": "Steady",
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": "Ch",
        "diabetesMed": "Yes",
        "diag_1": "250.8",
        "diag_2": "401.9",
        "diag_3": "428.0",
    }

    prediction_output = predictor.predict_patient_dict(raw_sample_patient, top_k_drivers=5)
    print("\nInference Verification Result for Sample High-Risk Patient Record:")
    print(json.dumps(prediction_output, indent=2))

    # Sanity Assertions for Phase 6 Handoff
    assert "readmission_risk_probability" in prediction_output
    assert "risk_band" in prediction_output
    assert prediction_output["risk_band"] in ["LOW", "MODERATE", "HIGH"]
    assert len(prediction_output["top_risk_drivers"]) > 0
    print("\nPhase 6 Handoff Pipeline Sanity Checks: ALL PASSED ✓")

    # Generate Final Execution Report
    generate_markdown_report(master_metrics, rag_spec, prediction_output)
    print("\n" + "=" * 80)
    print("✅ MASTER EXECUTION COMPLETE: PHASES 3–6 SUCCESSFULLY EXECUTED & FINALIZED")
    print("=" * 80)


def generate_markdown_report(metrics, rag_spec, sample_pred):
    report_path = "ml_engine/reports/model_evaluation/phase3_6_execution_report.md"
    content = f"""# AvertCare ML Engine — Phase 3 to Phase 6 Final Execution Report

## 1. Executive Summary
This report documents the end-to-end execution, evaluation, and serialization of Phases 3–6 of the AvertCare Hospital Readmission Risk Prediction system.

---

## 2. Phase 3: Model Performance Comparison

| Model | Val AUROC | Val AUPRC | Val F1 (Opt) | Test AUROC | Test AUPRC | Test F1 (Opt) | Opt Thresh |
|-------|-----------|-----------|--------------|------------|------------|---------------|------------|
| **Logistic Regression** | {metrics['Logistic_Regression']['val_default_0.5']['auroc']} | {metrics['Logistic_Regression']['val_default_0.5']['auprc']} | {metrics['Logistic_Regression']['val_optimal_thresh']['f1_score']} | {metrics['Logistic_Regression']['test_default_0.5']['auroc']} | {metrics['Logistic_Regression']['test_default_0.5']['auprc']} | {metrics['Logistic_Regression']['test_optimal_thresh']['f1_score']} | {metrics['Logistic_Regression']['val_optimal_thresh']['threshold']} |
| **Random Forest** | {metrics['Random_Forest']['val_default_0.5']['auroc']} | {metrics['Random_Forest']['val_default_0.5']['auprc']} | {metrics['Random_Forest']['val_optimal_thresh']['f1_score']} | {metrics['Random_Forest']['test_default_0.5']['auroc']} | {metrics['Random_Forest']['test_default_0.5']['auprc']} | {metrics['Random_Forest']['test_optimal_thresh']['f1_score']} | {metrics['Random_Forest']['val_optimal_thresh']['threshold']} |
| **FT-Transformer** | {metrics['FT_Transformer']['val_default_0.5']['auroc']} | {metrics['FT_Transformer']['val_default_0.5']['auprc']} | {metrics['FT_Transformer']['val_optimal_thresh']['f1_score']} | {metrics['FT_Transformer']['test_default_0.5']['auroc']} | {metrics['FT_Transformer']['test_default_0.5']['auprc']} | {metrics['FT_Transformer']['test_optimal_thresh']['f1_score']} | {metrics['FT_Transformer']['val_optimal_thresh']['threshold']} |

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
{json.dumps(sample_pred, indent=2)}
```
"""
    with open(report_path, "w") as f:
        f.write(content)
    print(f"Final report saved -> {report_path}")


if __name__ == "__main__":
    run_full_phase3_6_pipeline()
