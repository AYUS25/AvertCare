"""
Baseline Models Training Script (Logistic Regression & Random Forest)
Phase 3 Model Training on Tabular Preprocessed Features
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from evaluate import compute_metrics, find_optimal_threshold, plot_roc_curves, plot_pr_curves, save_metrics_summary


def train_and_evaluate_baselines():
    print("=" * 70)
    print("PHASE 3: TRAINING BASELINE MODELS (LOGISTIC REGRESSION & RANDOM FOREST)")
    print("=" * 70)

    # 1. Load Data
    data_dir = "ml_engine/data/processed"
    df_train = pd.read_csv(os.path.join(data_dir, "train_with_ids.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "val_with_ids.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "test_with_ids.csv"))

    target_col = "readmitted_binary"
    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    y_test = df_test[target_col].values

    # 2. Load Fitted Preprocessor (OHE)
    prep_obj = joblib.load(os.path.join(data_dir, "preprocessor_onehot.joblib"))
    preprocessor = prep_obj["preprocessor"] if isinstance(prep_obj, dict) else prep_obj

    print("Transforming feature sets with fitted preprocessor_onehot.joblib (NO refitting)...")
    X_train_trans = preprocessor.transform(df_train)
    X_val_trans = preprocessor.transform(df_val)
    X_test_trans = preprocessor.transform(df_test)

    print(f"X_train_trans shape: {X_train_trans.shape}")
    print(f"X_val_trans shape:   {X_val_trans.shape}")
    print(f"X_test_trans shape:  {X_test_trans.shape}")

    models_dir = "ml_engine/models"
    metrics_dir = "ml_engine/metrics"
    reports_dir = "ml_engine/reports/model_evaluation"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    all_metrics = {}
    val_probs = {}
    test_probs = {}

    # ---------------------------------------------------------
    # Model 1: Logistic Regression
    # ---------------------------------------------------------
    print("\n--- Model 1: Logistic Regression ---")
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    lr.fit(X_train_trans, y_train)

    p_val_lr = lr.predict_proba(X_val_trans)[:, 1]
    p_test_lr = lr.predict_proba(X_test_trans)[:, 1]
    val_probs["Logistic Regression"] = p_val_lr
    test_probs["Logistic Regression"] = p_test_lr

    opt_th_lr = find_optimal_threshold(y_val, p_val_lr, metric="f1")
    print(f"Logistic Regression optimal threshold (val set): {opt_th_lr:.3f}")

    metrics_val_lr = compute_metrics(y_val, p_val_lr, threshold=0.5)
    metrics_val_lr_opt = compute_metrics(y_val, p_val_lr, threshold=opt_th_lr)
    metrics_test_lr = compute_metrics(y_test, p_test_lr, threshold=0.5)
    metrics_test_lr_opt = compute_metrics(y_test, p_test_lr, threshold=opt_th_lr)

    all_metrics["Logistic_Regression"] = {
        "val_default_0.5": metrics_val_lr,
        "val_optimal_thresh": metrics_val_lr_opt,
        "test_default_0.5": metrics_test_lr,
        "test_optimal_thresh": metrics_test_lr_opt,
    }

    joblib.dump(lr, os.path.join(models_dir, "logistic_regression.joblib"))
    print(f"Saved Logistic Regression model -> {models_dir}/logistic_regression.joblib")
    print(f"Val AUROC: {metrics_val_lr['auroc']} | Val AUPRC: {metrics_val_lr['auprc']} | Val F1 (opt): {metrics_val_lr_opt['f1_score']}")
    print(f"Test AUROC: {metrics_test_lr['auroc']} | Test AUPRC: {metrics_test_lr['auprc']} | Test F1 (opt): {metrics_test_lr_opt['f1_score']}")

    # ---------------------------------------------------------
    # Model 2: Random Forest
    # ---------------------------------------------------------
    print("\n--- Model 2: Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_trans, y_train)

    p_val_rf = rf.predict_proba(X_val_trans)[:, 1]
    p_test_rf = rf.predict_proba(X_test_trans)[:, 1]
    val_probs["Random Forest"] = p_val_rf
    test_probs["Random Forest"] = p_test_rf

    opt_th_rf = find_optimal_threshold(y_val, p_val_rf, metric="f1")
    print(f"Random Forest optimal threshold (val set): {opt_th_rf:.3f}")

    metrics_val_rf = compute_metrics(y_val, p_val_rf, threshold=0.5)
    metrics_val_rf_opt = compute_metrics(y_val, p_val_rf, threshold=opt_th_rf)
    metrics_test_rf = compute_metrics(y_test, p_test_rf, threshold=0.5)
    metrics_test_rf_opt = compute_metrics(y_test, p_test_rf, threshold=opt_th_rf)

    all_metrics["Random_Forest"] = {
        "val_default_0.5": metrics_val_rf,
        "val_optimal_thresh": metrics_val_rf_opt,
        "test_default_0.5": metrics_test_rf,
        "test_optimal_thresh": metrics_test_rf_opt,
    }

    joblib.dump(rf, os.path.join(models_dir, "random_forest.joblib"))
    print(f"Saved Random Forest model -> {models_dir}/random_forest.joblib")
    print(f"Val AUROC: {metrics_val_rf['auroc']} | Val AUPRC: {metrics_val_rf['auprc']} | Val F1 (opt): {metrics_val_rf_opt['f1_score']}")
    print(f"Test AUROC: {metrics_test_rf['auroc']} | Test AUPRC: {metrics_test_rf['auprc']} | Test F1 (opt): {metrics_test_rf_opt['f1_score']}")

    # Save summary metrics
    save_metrics_summary(all_metrics, os.path.join(metrics_dir, "baseline_metrics.json"))
    print(f"\nSaved metrics summary -> {metrics_dir}/baseline_metrics.json")

    # Generate evaluation plots
    plot_roc_curves(test_probs, y_test, os.path.join(reports_dir, "baseline_roc_curves.png"))
    plot_pr_curves(test_probs, y_test, os.path.join(reports_dir, "baseline_pr_curves.png"))
    print(f"Generated ROC & PR curves -> {reports_dir}/")

    return all_metrics, test_probs


if __name__ == "__main__":
    train_and_evaluate_baselines()
