"""
run_phase1_2.py — Phase 1 & 2 Orchestrator
============================================
Executes:
  Phase 1 — Data acquisition, cleaning, target formulation, EDA plots.
  Phase 2 — Train/val/test split, preprocessor fitting, CSV/artifact export.

Usage
-----
  cd ml_engine/
  python run_phase1_2.py

Output files (all under ml_engine/data/processed/)
----------------------------------------------------
  train_with_ids.csv   — Training split (human-readable; includes encounter_id, patient_nbr)
  val_with_ids.csv     — Validation split (human-readable)
  test_with_ids.csv    — Test split (human-readable)
  split_manifest.csv   — encounter_id | patient_nbr | split (for M2 traceability)
  preprocessor_onehot.joblib  — Fitted ColumnTransformer for LR/RF
  preprocessor_ft.joblib      — Fitted ColumnTransformer for FT-Transformer
  DATA_DICTIONARY.md   — Precise documentation of every cleaning decision
  RAG_HANDOFF.md       — Interface contract between M1 and M2

EDA plots (under ml_engine/reports/eda/)
-----------------------------------------
  class_balance.png
  readmission_by_age.png
  readmission_by_diag1.png
  readmission_by_inpatient.png
  numeric_histograms.png

Notes on deferred preprocessing
---------------------------------
  The *_with_ids.csv files retain HUMAN-READABLE values (string categories,
  unscaled numerics) intentionally.  Numeric scaling (StandardScaler) and
  categorical encoding (OHE / OrdinalEncoder) are applied ONLY inside the
  Phase-3 sklearn pipeline via the saved .joblib preprocessors.

  encounter_id and patient_nbr are included in the CSVs ONLY for M2/RAG
  matching and audit traceability.  They are excluded from all model features.
"""

import os
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Add ml_engine directory to path so local imports work from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean import clean_data
from split import split_data, build_preprocessors


def _fetch_dataset() -> pd.DataFrame:
    """Download the UCI dataset (id=296) or fall back to a local raw CSV."""
    try:
        from ucimlrepo import fetch_ucirepo
        print("[data] Fetching dataset from UCI ML Repository (id=296)…")
        dataset = fetch_ucirepo(id=296)
        X = dataset.data.features
        y = dataset.data.targets
        ids = dataset.data.ids
        df = pd.concat([ids, X, y], axis=1)
        print(f"[data] Downloaded: {df.shape}")
        return df
    except Exception as exc:
        print(f"[data] UCI download failed: {exc}")
        raw_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "raw", "diabetic_data.csv"
        )
        if os.path.exists(raw_path):
            print(f"[data] Loading from local file: {raw_path}")
            df = pd.read_csv(raw_path)
            print(f"[data] Local file: {df.shape}")
            return df
        raise FileNotFoundError(
            f"Dataset not available. Download failed and local file missing: {raw_path}"
        ) from exc


def _eda_plots(df_clean: pd.DataFrame, numeric_cols: list, reports_dir: str) -> None:
    """Generate and save exploratory data analysis plots."""

    # Class balance
    fig, ax = plt.subplots()
    df_clean["readmitted_binary"].value_counts().plot(
        kind="bar", ax=ax, title="Class Balance (0=Not readmitted, 1=Readmitted <30 days)"
    )
    ax.set_xlabel("readmitted_binary")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(reports_dir, "class_balance.png"))
    plt.close(fig)

    # Readmission rate by age group
    if "age_numeric" in df_clean.columns:
        fig, ax = plt.subplots()
        df_clean.groupby("age_numeric")["readmitted_binary"].mean().plot(
            kind="bar", ax=ax, title="Readmission Rate by Age Midpoint"
        )
        ax.set_ylabel("Positive Rate")
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, "readmission_by_age.png"))
        plt.close(fig)

    # Readmission rate by primary diagnosis group
    if "diag_1_group" in df_clean.columns:
        fig, ax = plt.subplots()
        df_clean.groupby("diag_1_group")["readmitted_binary"].mean().plot(
            kind="bar", ax=ax, title="Readmission Rate by Primary Diagnosis Group"
        )
        ax.set_ylabel("Positive Rate")
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, "readmission_by_diag1.png"))
        plt.close(fig)

    # Readmission rate by prior inpatient bucket
    if "number_inpatient" in df_clean.columns:
        _tmp = df_clean.copy()
        _tmp["inpatient_bucket"] = pd.cut(
            _tmp["number_inpatient"],
            bins=[-1, 0, 1, 3, 10, 100],
            labels=["0", "1", "2-3", "4-10", ">10"],
        )
        fig, ax = plt.subplots()
        _tmp.groupby("inpatient_bucket", observed=True)["readmitted_binary"].mean().plot(
            kind="bar", ax=ax, title="Readmission Rate by Prior Inpatient Visits"
        )
        ax.set_ylabel("Positive Rate")
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, "readmission_by_inpatient.png"))
        plt.close(fig)

    # Numeric feature histograms
    if numeric_cols:
        axes = df_clean[numeric_cols].hist(figsize=(15, 10), bins=20)
        plt.suptitle("Numeric Feature Distributions", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "numeric_histograms.png"))
        plt.close("all")

    print(f"[eda] Plots saved to {reports_dir}")


def _write_data_dictionary(path: str, df_train: pd.DataFrame,
                            numeric_cols: list, categorical_cols: list) -> None:
    """Write a precise DATA_DICTIONARY.md based on the actual implementation."""
    lines = [
        "# Data Dictionary\n",
        "## Dataset\n",
        "Source: UCI ML Repository — Diabetes 130-US Hospitals (1999–2008), id=296.  \n",
        "Original: ~101,766 encounters × 50 features.  \n",
        "After cleaning: see row counts below.\n",
        "\n",
        "## Target\n",
        "| Original value | `readmitted_binary` |\n",
        "|---|---|\n",
        "| `<30` | **1** (positive: readmitted within 30 days) |\n",
        "| `>30` | 0 |\n",
        "| `NO`  | 0 |\n",
        "\n",
        "## Cleaning Steps (in order)\n",
        "1. `readmitted` → `readmitted_binary` (see Target above); original column dropped.\n",
        "2. All `?` values replaced with NaN.\n",
        "3. **`weight`**: dropped — >96% missing; not recoverable.\n",
        "4. **`discharge_disposition_id`** rows 11, 13, 14, 19, 20, 21 removed (death / hospice).\n",
        "5. **`gender == 'Unknown/Invalid'`** rows removed (< 5 rows).\n",
        "6. **`race`**: NaN → `'Unknown'` (2.2% of rows had missing race).\n",
        "7. **`payer_code`**: NaN → `'Unknown'` (39.7% missing — information-free but useful category).\n",
        "8. **`medical_specialty`**: NaN → `'Unknown'` (48.9% missing). Specialties representing\n",
        "   < 1% of encounters collapsed into `'Other'`.\n",
        "9. **`max_glu_serum`**: NaN → `'Not_Tested'` (94.8% not performed — test not ordered ≠ normal).\n",
        "   Sentinel `'Not_Tested'` is used instead of `'None'` because pandas.read_csv() treats\n",
        "   the string `'None'` as NaN by default, making it unsafe as a CSV sentinel.\n",
        "10. **`A1Cresult`**: NaN → `'Not_Tested'` (83.1% not performed — same rationale).\n",
        "11. **`admission_type_id`**, **`discharge_disposition_id`**, **`admission_source_id`**:\n",
        "    converted to string categories; values < 0.5% of encounters collapsed to `'Other'`.\n",
        "12. Constant/near-constant columns dropped automatically.\n",
        "13. **`diag_1`**, **`diag_2`**, **`diag_3`**: each ICD-9 code grouped into one of:\n",
        "    `Circulatory`, `Respiratory`, `Digestive`, `Genitourinary`, `Neoplasms`,\n",
        "    `Musculoskeletal`, `Injury`, `Diabetes`, `Other`. Original columns dropped.\n",
        "14. **`age`**: band string (e.g. `[60-70)`) → numeric midpoint (`age_numeric = 65`).\n",
        "15. Medication columns present as `No / Steady / Up / Down`. Columns where > 99% of\n",
        "    encounters are `'No'` are dropped (near-constant). Retained: see CSV columns.\n",
        "\n",
        "## Engineered Features\n",
        "| Feature | Formula |\n",
        "|---|---|\n",
        "| `total_prior_visits` | `number_outpatient + number_emergency + number_inpatient` |\n",
        "| `num_med_changes` | count of medication columns with value `'Up'` or `'Down'` |\n",
        "| `num_meds_active` | count of medication columns with value ≠ `'No'` |\n",
        "| `age_numeric` | numeric midpoint of the original 10-year age band |\n",
        "\n",
        "## ID Columns (MUST NOT be model features)\n",
        "| Column | Purpose |\n",
        "|---|---|\n",
        "| `encounter_id` | Unique encounter identifier — retained for M2/RAG matching only |\n",
        "| `patient_nbr` | Patient identifier — used to enforce patient-level split grouping |\n",
        "\n",
        "## Split Strategy\n",
        "- 80 / 10 / 10 (train / val / test)\n",
        "- Stratified by `readmitted_binary` (preserves class ratio in each split)\n",
        "- Grouped by `patient_nbr`: ALL encounters of one patient go to exactly ONE split\n",
        "- No patient overlap between any pair of splits (verified with assertions)\n",
        "- Random seed: 42 (fixed throughout the project)\n",
        "- Algorithm: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`\n",
        "\n",
        "## Class Imbalance\n",
        "- Positive class (`readmitted_binary=1`) ≈ 11.4% across all splits\n",
        "- Strategy: `class_weight='balanced'` applied to training estimators only\n",
        "- NO resampling (SMOTE etc.) is applied to validation or test data\n",
        "- If SMOTE is used in Phase 3, it is applied ONLY inside the training pipeline\n",
        "  after the train/val/test split, never before\n",
        "\n",
        "## Deferred Preprocessing (Phase 3 pipeline)\n",
        "The `*_with_ids.csv` files retain HUMAN-READABLE values (string categories,\n",
        "unscaled numerics) so M2 can use them directly for RAG.\n",
        "The following transformations are applied in the Phase-3 sklearn pipeline:\n",
        "- Numeric features: median imputation → StandardScaler\n",
        "- Categorical features: mode imputation → OneHotEncoder (LR/RF)\n",
        "  OR OrdinalEncoder (FT-Transformer)\n",
        "Fitted preprocessors are saved as:\n",
        "- `ml_engine/data/processed/preprocessor_onehot.joblib`\n",
        "- `ml_engine/data/processed/preprocessor_ft.joblib`\n",
        "\n",
        "## Numeric Features\n",
        f"  {', '.join(numeric_cols)}\n",
        "\n",
        "## Categorical Features\n",
        f"  {', '.join(categorical_cols)}\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[docs] DATA_DICTIONARY.md written to {path}")


def _write_rag_handoff(path: str) -> None:
    """Write RAG_HANDOFF.md with explicit leakage-prevention requirements."""
    content = """\
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
"""
    with open(path, "w") as f:
        f.write(content)
    print(f"[docs] RAG_HANDOFF.md written to {path}")


def main() -> None:
    """Run Phases 1 and 2 end-to-end."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REPORTS_DIR   = os.path.join(BASE_DIR, "ml_engine", "reports", "eda")
    DATA_PROC_DIR = os.path.join(BASE_DIR, "ml_engine", "data", "processed")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DATA_PROC_DIR, exist_ok=True)

    # ── PHASE 1: Acquire & Clean ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: DATA ACQUISITION & CLEANING")
    print("=" * 60)
    df_raw = _fetch_dataset()
    print(f"[data] Raw shape: {df_raw.shape}")
    print(f"[data] Columns: {df_raw.columns.tolist()}")

    df_clean = clean_data(df_raw)

    # Identify column roles (exclude IDs and target from feature lists)
    IDENT_COLS  = ["encounter_id", "patient_nbr"]
    TARGET_COL  = "readmitted_binary"

    numeric_cols = (
        df_clean.select_dtypes(include=[np.number]).columns
        .difference(IDENT_COLS + [TARGET_COL])
        .tolist()
    )
    categorical_cols = (
        df_clean.select_dtypes(exclude=[np.number]).columns
        .difference(IDENT_COLS + ["split"])
        .tolist()
    )

    print(f"\n[features] Numeric ({len(numeric_cols)}): {numeric_cols}")
    print(f"[features] Categorical ({len(categorical_cols)}): {categorical_cols}")

    # EDA plots
    _eda_plots(df_clean, numeric_cols, REPORTS_DIR)

    # ── PHASE 2: Split, Preprocess, Export ───────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: SPLIT & PREPROCESSOR CONSTRUCTION")
    print("=" * 60)
    df_train, df_val, df_test = split_data(df_clean, random_state=42)

    # Final statistics
    total = len(df_train) + len(df_val) + len(df_test)
    print(f"\n[stats] Total rows: {total}")
    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        pos = split[TARGET_COL].sum()
        pct = pos / len(split) * 100
        print(f"[stats] {name}: {len(split)} rows | pos={pos} ({pct:.2f}%) "
              f"| neg={len(split)-pos} | patients={split['patient_nbr'].nunique()}")

    # Patient overlap (must all be 0)
    train_pts = set(df_train["patient_nbr"])
    val_pts   = set(df_val["patient_nbr"])
    test_pts  = set(df_test["patient_nbr"])
    print(f"\n[leakage] Train ∩ Val  overlap: {len(train_pts & val_pts)}")
    print(f"[leakage] Train ∩ Test overlap: {len(train_pts & test_pts)}")
    print(f"[leakage] Val   ∩ Test overlap: {len(val_pts & test_pts)}")

    # Build and persist preprocessors (fitted on TRAIN only)
    prep_ohe, prep_ft = build_preprocessors(
        df_train=df_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        output_dir=DATA_PROC_DIR,
    )

    # ── Export human-readable CSVs ────────────────────────────────────────────
    print("\n[export] Writing CSVs…")
    df_train.to_csv(os.path.join(DATA_PROC_DIR, "train_with_ids.csv"), index=False)
    df_val.to_csv(  os.path.join(DATA_PROC_DIR, "val_with_ids.csv"),   index=False)
    df_test.to_csv( os.path.join(DATA_PROC_DIR, "test_with_ids.csv"),  index=False)

    split_manifest = pd.concat([
        df_train[["encounter_id", "patient_nbr", "split"]],
        df_val  [["encounter_id", "patient_nbr", "split"]],
        df_test [["encounter_id", "patient_nbr", "split"]],
    ])
    split_manifest.to_csv(os.path.join(DATA_PROC_DIR, "split_manifest.csv"), index=False)
    print(f"[export] CSVs written to {DATA_PROC_DIR}")

    # ── Verify CSV round-trip (sentinel check) ────────────────────────────────
    print("\n[verify] Checking CSV round-trip for critical sentinels…")
    df_train_rt = pd.read_csv(os.path.join(DATA_PROC_DIR, "train_with_ids.csv"))
    for col in ["max_glu_serum", "A1Cresult"]:
        nan_count = df_train_rt[col].isna().sum()
        nt_count  = (df_train_rt[col] == "Not_Tested").sum()
        print(f"  {col}: NaN={nan_count}, 'Not_Tested'={nt_count} → "
              f"{'OK' if nan_count == 0 else 'BUG: still NaN!'}")

    # ── Documentation ─────────────────────────────────────────────────────────
    _write_data_dictionary(
        os.path.join(DATA_PROC_DIR, "DATA_DICTIONARY.md"),
        df_train, numeric_cols, categorical_cols,
    )
    _write_rag_handoff(os.path.join(DATA_PROC_DIR, "RAG_HANDOFF.md"))

    print("\n" + "=" * 60)
    print("PHASE 1 & 2 COMPLETE")
    print("=" * 60)
    print("Artifacts created:")
    for f in ["train_with_ids.csv", "val_with_ids.csv", "test_with_ids.csv",
              "split_manifest.csv", "preprocessor_onehot.joblib",
              "preprocessor_ft.joblib", "DATA_DICTIONARY.md", "RAG_HANDOFF.md"]:
        fp = os.path.join(DATA_PROC_DIR, f)
        exists = os.path.exists(fp)
        size_kb = os.path.getsize(fp) // 1024 if exists else 0
        print(f"  {'✓' if exists else '✗'} {f} ({size_kb} KB)")


if __name__ == "__main__":
    main()
