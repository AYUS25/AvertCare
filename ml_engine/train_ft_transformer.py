"""
FT-Transformer Training Script
Phase 3 Model Training on FT Tabular Embeddings
"""

import os
import json
import joblib
import torch
import numpy as np
import pandas as pd

from ft_transformer import FTTransformerClassifier
from evaluate import compute_metrics, find_optimal_threshold, plot_roc_curves, plot_pr_curves, save_metrics_summary


def train_and_evaluate_ft_transformer():
    print("=" * 70)
    print("PHASE 3: TRAINING FT-TRANSFORMER (TABULAR DEEP LEARNING)")
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

    # 2. Load Fitted Preprocessor (FT)
    pre_ft = joblib.load(os.path.join(data_dir, "preprocessor_ft.joblib"))

    print("Transforming feature sets with fitted preprocessor_ft.joblib (NO refitting)...")
    X_train_trans = pre_ft.transform(df_train)
    X_val_trans = pre_ft.transform(df_val)
    X_test_trans = pre_ft.transform(df_test)

    print(f"X_train_trans shape: {X_train_trans.shape}")
    print(f"X_val_trans shape:   {X_val_trans.shape}")
    print(f"X_test_trans shape:  {X_test_trans.shape}")

    # Compute cardinalities for the 22 categorical columns (indices 12..33) with +1 shift for unknown category (0)
    X_all = np.vstack([X_train_trans, X_val_trans, X_test_trans])
    X_all_cat = np.maximum(0, X_all[:, 12:].astype(int) + 1)
    cat_cardinalities = [int(X_all_cat[:, i].max() + 1) for i in range(22)]
    print(f"Categorical feature cardinalities: {cat_cardinalities}")

    models_dir = "ml_engine/models"
    metrics_dir = "ml_engine/metrics"
    reports_dir = "ml_engine/reports/model_evaluation"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 3. Instantiate & Train FT-Transformer
    print("\nTraining FT-Transformer PyTorch Model...")
    ft_model = FTTransformerClassifier(
        n_num_features=12,
        cat_cardinalities=cat_cardinalities,
        d_token=64,
        n_layers=3,
        n_heads=4,
        d_ff=128,
        dropout=0.1,
        lr=1e-3,
        batch_size=256,
        epochs=15,
        patience=4,
    )

    ft_model.fit(X_train_trans, y_train, eval_set=(X_val_trans, y_val))

    # 4. Predict & Evaluate
    p_val = ft_model.predict_proba(X_val_trans)[:, 1]
    p_test = ft_model.predict_proba(X_test_trans)[:, 1]

    opt_th = find_optimal_threshold(y_val, p_val, metric="f1")
    print(f"\nFT-Transformer optimal threshold (val set): {opt_th:.3f}")

    metrics_val = compute_metrics(y_val, p_val, threshold=0.5)
    metrics_val_opt = compute_metrics(y_val, p_val, threshold=opt_th)
    metrics_test = compute_metrics(y_test, p_test, threshold=0.5)
    metrics_test_opt = compute_metrics(y_test, p_test, threshold=opt_th)

    metrics_summary = {
        "FT_Transformer": {
            "val_default_0.5": metrics_val,
            "val_optimal_thresh": metrics_val_opt,
            "test_default_0.5": metrics_test,
            "test_optimal_thresh": metrics_test_opt,
        }
    }

    # Save artifacts
    torch.save(ft_model.model.state_dict(), os.path.join(models_dir, "ft_transformer.pt"))
    joblib.dump(ft_model, os.path.join(models_dir, "ft_transformer.joblib"))
    print(f"Saved FT-Transformer model weights -> {models_dir}/ft_transformer.pt")
    print(f"Saved FT-Transformer classifier object -> {models_dir}/ft_transformer.joblib")

    print(f"Val AUROC: {metrics_val['auroc']} | Val AUPRC: {metrics_val['auprc']} | Val F1 (opt): {metrics_val_opt['f1_score']}")
    print(f"Test AUROC: {metrics_test['auroc']} | Test AUPRC: {metrics_test['auprc']} | Test F1 (opt): {metrics_test_opt['f1_score']}")

    save_metrics_summary(metrics_summary, os.path.join(metrics_dir, "ft_transformer_metrics.json"))
    print(f"Saved metrics summary -> {metrics_dir}/ft_transformer_metrics.json")

    return metrics_summary, p_test


if __name__ == "__main__":
    train_and_evaluate_ft_transformer()
