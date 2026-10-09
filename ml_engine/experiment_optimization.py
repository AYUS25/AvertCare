"""
experiment_optimization.py — Deep Feature Engineering & Hyperparameter Tuning
================================================================================
Grid search & evaluation of:
- Clinical interaction features
- Fine-grained leakage-free RAG historical rate features (single & combination keys)
- Logistic Regression, Random Forest, FT-Transformer, and XGBoost
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean import clean_data
from split import build_preprocessors

DATA_DIR = "ml_engine/data/processed"
RAG_CSV = "data/processed/train_with_rag.csv"

def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds clinically meaningful features derived strictly from encounter data at prediction time."""
    df = df.copy()
    time_hosp = df["time_in_hospital"].clip(lower=1)
    
    # Ratios / Intensities
    df["labs_per_day"] = df["num_lab_procedures"] / time_hosp
    df["meds_per_day"] = df["num_medications"] / time_hosp
    df["procedures_per_day"] = df["num_procedures"] / time_hosp
    df["diagnoses_per_day"] = df["number_diagnoses"] / time_hosp
    df["lab_med_ratio"] = df["num_lab_procedures"] / (df["num_medications"] + 1.0)
    
    # Flags & Buckets
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["has_prior_emergency"] = (df["number_emergency"] > 0).astype(int)
    df["has_prior_outpatient"] = (df["number_outpatient"] > 0).astype(int)
    
    tot_visits = df["total_prior_visits"].clip(lower=0) + 1.0
    df["inpatient_ratio"] = df["number_inpatient"] / tot_visits
    df["emergency_ratio"] = df["number_emergency"] / tot_visits
    
    df["high_utilizer"] = (df["total_prior_visits"] >= 3).astype(int)
    df["polypharmacy"] = (df["num_medications"] >= 10).astype(int)
    df["clinical_complexity"] = df["num_lab_procedures"] + (df["num_procedures"] * 2.5) + (df["number_diagnoses"] * 3.0)
    
    # Non-linear terms & Interactions
    df["inpatient_sq"] = df["number_inpatient"] ** 2
    df["emergency_sq"] = df["number_emergency"] ** 2
    df["age_inpatient_interaction"] = df["age_numeric"] * (df["number_inpatient"] + 1.0)
    df["age_diag_interaction"] = df["age_numeric"] * (df["number_diagnoses"] + 1.0)
    
    active_meds = df["num_meds_active"].clip(lower=0) + 1.0
    df["med_change_ratio"] = df["num_med_changes"] / active_meds
    
    # Combination category strings for target encoding
    df["diag_disposition_combo"] = df["diag_1_group"].astype(str) + "_" + df["discharge_disposition_id"].astype(str)
    df["admission_disposition_combo"] = df["admission_type_id"].astype(str) + "_" + df["discharge_disposition_id"].astype(str)
    
    return df

def build_enriched_rag_features(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """
    Builds leakage-free historical readmission rate features derived strictly from TRAIN set.
    Uses Out-of-Fold (OOF) encoding for Train set, and direct train-mapping for Val & Test sets.
    """
    df_tr = df_train.copy()
    df_vl = df_val.copy()
    df_te = df_test.copy()
    
    target_col = "readmitted_binary"
    global_mean = df_tr[target_col].mean()
    
    cat_cols = [
        "diag_1_group", "discharge_disposition_id", "admission_type_id",
        "medical_specialty", "diag_disposition_combo", "admission_disposition_combo"
    ]
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in cat_cols:
        col_name = f"rag_hist_rate_{col}"
        df_tr[col_name] = global_mean
        
        for tr_idx, val_idx in kf.split(df_tr):
            tr_fold = df_tr.iloc[tr_idx]
            stats = tr_fold.groupby(col)[target_col].agg(["count", "mean"])
            m = 15.0  # smoothing parameter
            smoothed = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
            mapped_vals = df_tr.iloc[val_idx][col].map(smoothed).fillna(global_mean)
            df_tr.iloc[val_idx, df_tr.columns.get_loc(col_name)] = mapped_vals
            
        full_stats = df_tr.groupby(col)[target_col].agg(["count", "mean"])
        m = 15.0
        full_smoothed = (full_stats["count"] * full_stats["mean"] + m * global_mean) / (full_stats["count"] + m)
        
        df_vl[col_name] = df_vl[col].map(full_smoothed).fillna(global_mean)
        df_te[col_name] = df_te[col].map(full_smoothed).fillna(global_mean)
        
    return df_tr, df_vl, df_te

def run_tuning():
    print("Loading datasets...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train_with_ids.csv"))
    df_val   = pd.read_csv(os.path.join(DATA_DIR, "val_with_ids.csv"))
    df_test  = pd.read_csv(os.path.join(DATA_DIR, "test_with_ids.csv"))
    
    rag_lookup = pd.read_csv(RAG_CSV)[["encounter_id", "rag_readmit_rate"]].drop_duplicates(subset="encounter_id")
    
    for df in [df_train, df_val, df_test]:
        df["rag_readmit_rate"] = df.merge(rag_lookup, on="encounter_id", how="left")["rag_readmit_rate"].fillna(0.0).values
        
    df_train_eng = add_clinical_features(df_train)
    df_val_eng   = add_clinical_features(df_val)
    df_test_eng  = add_clinical_features(df_test)
    
    df_tr_rag, df_vl_rag, df_te_rag = build_enriched_rag_features(df_train_eng, df_val_eng, df_test_eng)
    
    IDENT_COLS = ["encounter_id", "patient_nbr"]
    TARGET_COL = "readmitted_binary"
    COMBO_COLS = ["diag_disposition_combo", "admission_disposition_combo"]
    RAG_COLS   = ["rag_readmit_rate"] + [c for c in df_tr_rag.columns if c.startswith("rag_hist_rate_")]
    
    numeric_cols = df_tr_rag.select_dtypes(include=[np.number]).columns.difference(IDENT_COLS + [TARGET_COL] + RAG_COLS + ["split"]).tolist()
    categorical_cols = df_tr_rag.select_dtypes(exclude=[np.number]).columns.difference(IDENT_COLS + ["split"] + COMBO_COLS).tolist()
    
    print(f"Engineered Numeric ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")
    print(f"RAG Columns ({len(RAG_COLS)}): {RAG_COLS}")
    
    prep_ohe, prep_ft = build_preprocessors(df_tr_rag, numeric_cols, categorical_cols, output_dir="/tmp/prep_test2")
    
    X_tr_tab = prep_ohe.transform(df_tr_rag)
    X_vl_tab = prep_ohe.transform(df_vl_rag)
    
    X_tr_rag = np.hstack([X_tr_tab, df_tr_rag[RAG_COLS].values])
    X_vl_rag = np.hstack([X_vl_tab, df_vl_rag[RAG_COLS].values])
    
    y_tr = df_tr_rag[TARGET_COL].values
    y_vl = df_vl_rag[TARGET_COL].values
    
    print("\n--- TUNING XGBOOST ---")
    scale_pos = (len(y_tr) - sum(y_tr)) / sum(y_tr)
    
    best_xgb_auc = 0
    best_xgb_params = None
    
    for max_depth in [4, 5, 6]:
        for lr in [0.02, 0.03, 0.05]:
            for subsample in [0.7, 0.8]:
                for colsample in [0.6, 0.7]:
                    xgb = XGBClassifier(
                        n_estimators=400,
                        max_depth=max_depth,
                        learning_rate=lr,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        scale_pos_weight=scale_pos,
                        random_state=42,
                        n_jobs=-1,
                        early_stopping_rounds=30
                    )
                    xgb.fit(X_tr_rag, y_tr, eval_set=[(X_vl_rag, y_vl)], verbose=False)
                    p_vl = xgb.predict_proba(X_vl_rag)[:, 1]
                    auc = roc_auc_score(y_vl, p_vl)
                    if auc > best_xgb_auc:
                        best_xgb_auc = auc
                        best_xgb_params = {"max_depth": max_depth, "lr": lr, "subsample": subsample, "colsample": colsample, "n_est": xgb.best_iteration}
                        print(f"  New best XGB Val AUROC: {auc:.4f} (AUPRC: {average_precision_score(y_vl, p_vl):.4f}) with {best_xgb_params}")

    print("\n--- TUNING RANDOM FOREST ---")
    best_rf_auc = 0
    for max_depth in [14, 16, 18, 20]:
        for min_leaf in [2, 4, 8]:
            for cw in ["balanced", "balanced_subsample"]:
                rf = RandomForestClassifier(
                    n_estimators=250,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    class_weight=cw,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(X_tr_rag, y_tr)
                p_vl = rf.predict_proba(X_vl_rag)[:, 1]
                auc = roc_auc_score(y_vl, p_vl)
                if auc > best_rf_auc:
                    best_rf_auc = auc
                    print(f"  New best RF Val AUROC: {auc:.4f} (depth={max_depth}, min_leaf={min_leaf}, cw={cw})")

if __name__ == "__main__":
    run_tuning()
