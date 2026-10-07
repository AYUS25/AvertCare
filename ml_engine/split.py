"""
split.py — Phase 2: Train/Val/Test Splitting & Preprocessor Construction
=========================================================================
Splitting strategy
------------------
  80 % train / 10 % validation / 10 % test
  Stratified by readmitted_binary (preserves ~11 % positive rate in each split).
  Grouped by patient_nbr: ALL encounters of one patient go to exactly ONE split,
  ensuring ZERO patient-level leakage across train / val / test.

  Implementation uses StratifiedGroupKFold(n_splits=10, shuffle=True, seed=42).
  Each fold produces a disjoint 10 % "test" slice.  We designate:
    - fold 0's test slice  → final TEST set
    - fold 1's test slice  → final VALIDATION set
    - folds 2–9's test slices (8 × 10 % = 80 %) → final TRAIN set
  Because StratifiedGroupKFold enforces group integrity, all patient_nbr values
  appear in exactly one of these partitions.

Preprocessor construction
--------------------------
  Two sklearn ColumnTransformer objects are built and fitted on TRAIN data only:
    preprocessor_onehot  — for Logistic Regression / Random Forest
    preprocessor_ft      — for FT-Transformer (OrdinalEncoder integers)

  Both are serialised with joblib so Phase 3 can load them without retraining.

  The exported *_with_ids.csv files remain in human-readable form (string
  categories, unscaled numerics) for M2/RAG traceability.  Numeric scaling and
  categorical encoding happen ONLY inside the Phase-3 sklearn pipeline.

ID columns
----------
  encounter_id and patient_nbr are exported in the CSVs for M2 matching and
  audit traceability.  They are EXCLUDED from the ColumnTransformer's
  numeric_cols and categorical_cols and must NOT be used as model features.
"""

import os
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


def split_data(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train / validation / test with patient-level group integrity.

    Parameters
    ----------
    df           : Cleaned DataFrame (output of clean_data).
    random_state : Random seed (default 42; fixed throughout the project).

    Returns
    -------
    (df_train, df_val, df_test) — each with a 'split' column added.

    Post-conditions (asserted internally)
    --------------------------------------
    - train ∩ val patient_nbr  = ∅
    - train ∩ test patient_nbr = ∅
    - val   ∩ test patient_nbr = ∅
    - target rate is approximately equal across all three splits.
    """
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)

    groups = df["patient_nbr"]
    y = df["readmitted_binary"]
    X = df.drop(columns=["readmitted_binary"])

    # Materialise all 10 folds: folds[i] = (train_idx, test_idx)
    # Each test_idx is a disjoint 10 % chunk; together they cover 100 % of rows.
    folds = list(sgkf.split(X, y, groups))

    test_idx  = folds[0][1]          # 10 % — held-out test
    val_idx   = folds[1][1]          # 10 % — held-out validation
    train_idx: List[int] = []
    for i in range(2, 10):           # 80 % — training
        train_idx.extend(folds[i][1])

    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    df_train["split"] = "train"
    df_val["split"]   = "val"
    df_test["split"]  = "test"

    # ── Leakage assertions ──────────────────────────────────────────────────
    train_pts = set(df_train["patient_nbr"])
    val_pts   = set(df_val["patient_nbr"])
    test_pts  = set(df_test["patient_nbr"])
    assert train_pts.isdisjoint(val_pts),  "LEAKAGE: Train and Val share patients!"
    assert train_pts.isdisjoint(test_pts), "LEAKAGE: Train and Test share patients!"
    assert val_pts.isdisjoint(test_pts),   "LEAKAGE: Val and Test share patients!"

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        n = len(split)
        pos = split["readmitted_binary"].sum()
        print(
            f"[split] {name}: {n} rows ({n/(len(df)):.1%}), "
            f"pos={pos} ({pos/n:.2%}), "
            f"unique patients={split['patient_nbr'].nunique()}"
        )

    return df_train, df_val, df_test


def build_preprocessors(
    df_train: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    output_dir: str = ".",
) -> Tuple[ColumnTransformer, ColumnTransformer]:
    """
    Fit two ColumnTransformer preprocessors on *df_train* and save them.

    Parameters
    ----------
    df_train        : Training split (human-readable, no encoding applied yet).
    numeric_cols    : Feature columns with numeric dtype (excluding IDs and target).
    categorical_cols: Feature columns with object/string dtype (excluding IDs).
    output_dir      : Directory where .joblib files are written.

    Returns
    -------
    (preprocessor_onehot, preprocessor_ft)
      preprocessor_onehot : For Logistic Regression & Random Forest.
                            Numeric → median impute → StandardScaler.
                            Categorical → mode impute → OneHotEncoder.
      preprocessor_ft     : For FT-Transformer.
                            Numeric → median impute → StandardScaler.
                            Categorical → mode impute → OrdinalEncoder (integers).

    Saved files
    -----------
    <output_dir>/preprocessor_onehot.joblib
    <output_dir>/preprocessor_ft.joblib

    IMPORTANT: These are fitted on TRAIN data ONLY.  Validation and test splits
    must be transformed with the same fitted objects, never refitted.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Numeric sub-pipeline: impute then scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Categorical sub-pipeline for OHE (LR / RF)
    # sparse_output=False → dense array, compatible with all downstream estimators.
    cat_pipeline_ohe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor_onehot = ColumnTransformer(
        transformers=[
            ("num", num_pipeline,    numeric_cols),
            ("cat", cat_pipeline_ohe, categorical_cols),
        ],
        remainder="drop",
    )

    # Categorical sub-pipeline for OrdinalEncoder (FT-Transformer)
    # unknown_value=-1 is later remapped to max_cardinality inside the Torch dataset.
    cat_pipeline_ft = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor_ft = ColumnTransformer(
        transformers=[
            ("num", num_pipeline,   numeric_cols),
            ("cat", cat_pipeline_ft, categorical_cols),
        ],
        remainder="drop",
    )

    # Fit on TRAIN only
    preprocessor_onehot.fit(df_train[numeric_cols + categorical_cols])
    preprocessor_ft.fit(df_train[numeric_cols + categorical_cols])

    # Persist to disk
    ohe_path = os.path.join(output_dir, "preprocessor_onehot.joblib")
    ft_path  = os.path.join(output_dir, "preprocessor_ft.joblib")
    joblib.dump(preprocessor_onehot, ohe_path)
    joblib.dump(preprocessor_ft,     ft_path)
    print(f"[split] Saved preprocessor_onehot → {ohe_path}")
    print(f"[split] Saved preprocessor_ft     → {ft_path}")

    return preprocessor_onehot, preprocessor_ft
