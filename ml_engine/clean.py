"""
clean.py — Phase 1: Data Cleaning & Target Formulation
=======================================================
Handles:
  - Target creation (readmitted_binary)
  - "?" → NaN replacement
  - Column drops (weight, constant cols)
  - Missing-value treatment (explicit sentinels that survive CSV round-trips)
  - Death/hospice removal
  - ICD-9 diagnosis grouping
  - Age band → numeric midpoint
  - Medication column pruning (>99% "No")
  - Engineered features (total_prior_visits, num_med_changes, num_meds_active)

IMPORTANT — sentinel choice for max_glu_serum / A1Cresult:
  pandas.read_csv() treats the string "None" as NaN by default (it is in the
  default na_values list).  Using "None" as a fill value is therefore INCORRECT
  because the sentinel is silently lost on every CSV round-trip.
  We use "Not_Tested" instead, which is never in the pandas na_values list and
  is semantically accurate (the test was simply not performed for that encounter).
"""

import pandas as pd
import numpy as np


# ICD-9 category mapping — aligned with Clore et al. (2014) grouping scheme
def _map_icd9(val: object) -> str:
    """Map a raw ICD-9 code to a clinically meaningful category string."""
    if pd.isna(val):
        return "Other"
    val_str = str(val).strip()
    if val_str in ("Unknown", "nan", ""):
        return "Other"
    # Diabetes (250.xx)
    if val_str.startswith("250"):
        return "Diabetes"
    # V-codes (supplementary) and E-codes (external causes)
    if val_str.startswith("V") or val_str.startswith("E"):
        return "Other"
    try:
        num = float(val_str)
    except ValueError:
        return "Other"

    if (390 <= num <= 459) or num == 785:
        return "Circulatory"
    elif (460 <= num <= 519) or num == 786:
        return "Respiratory"
    elif (520 <= num <= 579) or num == 787:
        return "Digestive"
    elif (580 <= num <= 629) or num == 788:
        return "Genitourinary"
    elif 140 <= num <= 239:
        return "Neoplasms"
    elif 710 <= num <= 739:
        return "Musculoskeletal"
    elif 800 <= num <= 999:
        return "Injury"
    else:
        return "Other"


def _age_midpoint(val: object) -> float:
    """Convert an age-band string like '[60-70)' to its numeric midpoint (65)."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip("[]() ")
    parts = val_str.split("-")
    if len(parts) == 2:
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except ValueError:
            return np.nan
    return np.nan


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute all Phase-1 cleaning steps and return a clean DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw concatenated DataFrame from ucimlrepo (ids + features + targets).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with readmitted_binary target, engineered features,
        and human-readable category strings (no numeric encoding yet — that is
        deferred to the Phase-3 sklearn pipeline so handoff CSVs stay readable).

    Missing-value treatment summary
    --------------------------------
    weight           : dropped entirely (>96% missing — not recoverable).
    payer_code       : "?" → NaN → "Unknown" (39.7% missing; information-free,
                       kept as explicit Unknown category).
    medical_specialty: "?" → NaN → "Unknown" (48.9% missing; same rationale).
                       Rare specialties (<1% each) collapsed to "Other".
    race             : "?" → NaN → "Unknown" (2.2% missing).
    max_glu_serum    : "?" → NaN → "Not_Tested" (94.8% not performed; this is
                       clinically informative — test not ordered ≠ normal result).
    A1Cresult        : "?" → NaN → "Not_Tested" (83.1% not performed; same).
    diag_1/2/3       : NaN → "Other" inside the ICD-9 grouping function.
    age              : band string → numeric midpoint; no missing in this dataset.
    Constant cols    : dropped automatically.
    Gender "Unknown/Invalid": rows removed (<5 rows).
    Death/hospice discharge IDs 11,13,14,19,20,21: rows removed (~2400 rows).
    """
    df = df.copy()
    print(f"[clean] Input shape: {df.shape}")

    # ── 1. Target ────────────────────────────────────────────────────────────
    if "readmitted" in df.columns:
        # <30  → 1 (positive: readmitted within 30 days)
        # >30  → 0
        # NO   → 0
        df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
        df.drop(columns=["readmitted"], inplace=True)
        pos = df["readmitted_binary"].sum()
        neg = len(df) - pos
        print(f"[clean] Target: pos={pos} ({pos/len(df):.2%}), neg={neg} ({neg/len(df):.2%})")

    # ── 2. Replace "?" sentinel with NaN ─────────────────────────────────────
    df.replace("?", np.nan, inplace=True)

    # ── 3. Drop weight (>96% missing) ────────────────────────────────────────
    if "weight" in df.columns:
        pct_missing = df["weight"].isna().mean()
        print(f"[clean] Dropping 'weight' ({pct_missing:.1%} missing).")
        df.drop(columns=["weight"], inplace=True)

    # ── 4. Remove death/hospice discharges ───────────────────────────────────
    if "discharge_disposition_id" in df.columns:
        df["discharge_disposition_id"] = pd.to_numeric(
            df["discharge_disposition_id"], errors="coerce"
        )
        death_ids = {11, 13, 14, 19, 20, 21}
        mask = df["discharge_disposition_id"].isin(death_ids)
        print(f"[clean] Removing {mask.sum()} death/hospice encounters.")
        df = df[~mask].copy()

    # ── 5. Remove Unknown/Invalid gender rows ────────────────────────────────
    if "gender" in df.columns:
        bad = (df["gender"] == "Unknown/Invalid").sum()
        if bad:
            print(f"[clean] Removing {bad} rows with gender='Unknown/Invalid'.")
        df = df[df["gender"] != "Unknown/Invalid"].copy()

    # ── 6. Fill missings with explicit, CSV-safe sentinels ───────────────────
    # CRITICAL: Do NOT use "None" — pandas.read_csv treats it as NaN by default.
    # Use "Unknown" for identity-like fields, "Not_Tested" for lab tests.

    if "race" in df.columns:
        df["race"] = df["race"].fillna("Unknown")

    if "payer_code" in df.columns:
        df["payer_code"] = df["payer_code"].fillna("Unknown")

    if "medical_specialty" in df.columns:
        df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
        # Collapse rare specialties (<1% of encounters) → "Other"
        spec_counts = df["medical_specialty"].value_counts(normalize=True)
        valid_specs = spec_counts[spec_counts >= 0.01].index
        df["medical_specialty"] = df["medical_specialty"].where(
            df["medical_specialty"].isin(valid_specs), "Other"
        )

    # max_glu_serum and A1Cresult: use "Not_Tested" (safe CSV-round-trip sentinel)
    for col in ["max_glu_serum", "A1Cresult"]:
        if col in df.columns:
            df[col] = df[col].fillna("Not_Tested")
            # Guard against any leftover string representations
            df[col] = df[col].replace({"nan": "Not_Tested", "None": "Not_Tested", "": "Not_Tested"})

    # ── 7. Admission/discharge/source IDs → categorical strings ──────────────
    # Group rare categories (<0.5%) into "Other" to avoid cardinality explosion.
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if col in df.columns:
            # Convert float remainders (e.g. 1.0 → "1")
            df[col] = df[col].apply(
                lambda x: str(int(x)) if pd.notnull(x) and isinstance(x, float) and x.is_integer()
                else str(x)
            )
            counts = df[col].value_counts(normalize=True)
            valid = counts[counts >= 0.005].index
            df[col] = df[col].where(df[col].isin(valid), "Other")

    # ── 8. Drop constant/near-constant columns ───────────────────────────────
    dropped_const = []
    for col in list(df.columns):
        if df[col].nunique() <= 1:
            dropped_const.append(col)
            df.drop(columns=[col], inplace=True)
    if dropped_const:
        print(f"[clean] Dropped constant columns: {dropped_const}")

    # ── 9. ICD-9 diagnosis grouping ──────────────────────────────────────────
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_group"] = df[col].apply(_map_icd9)
            df.drop(columns=[col], inplace=True)

    # ── 10. Age → numeric midpoint ────────────────────────────────────────────
    if "age" in df.columns:
        df["age_numeric"] = df["age"].apply(_age_midpoint)
        df.drop(columns=["age"], inplace=True)

    # ── 11. Engineered features ───────────────────────────────────────────────
    df["total_prior_visits"] = (
        df.get("number_outpatient", pd.Series(0, index=df.index))
        + df.get("number_emergency", pd.Series(0, index=df.index))
        + df.get("number_inpatient", pd.Series(0, index=df.index))
    )

    potential_med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
        "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
        "tolazamide", "examide", "citoglipton", "insulin",
        "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    med_cols = [c for c in potential_med_cols if c in df.columns]

    if med_cols:
        df["num_med_changes"] = (df[med_cols].isin(["Up", "Down"])).sum(axis=1)
        df["num_meds_active"] = (df[med_cols] != "No").sum(axis=1)

        # Drop individual medication columns that are >99% "No" (near-constant)
        dropped_meds = []
        for col in med_cols:
            pct_no = (df[col] == "No").mean()
            if pct_no > 0.99:
                dropped_meds.append(col)
                df.drop(columns=[col], inplace=True)
        if dropped_meds:
            print(f"[clean] Dropped medication columns (>99% 'No'): {dropped_meds}")

    # Clinical ratios, intensities & interaction features
    time_hosp = df.get("time_in_hospital", pd.Series(1, index=df.index)).clip(lower=1)
    df["labs_per_day"] = df.get("num_lab_procedures", 0) / time_hosp
    df["meds_per_day"] = df.get("num_medications", 0) / time_hosp
    df["procedures_per_day"] = df.get("num_procedures", 0) / time_hosp
    df["diagnoses_per_day"] = df.get("number_diagnoses", 0) / time_hosp
    df["lab_med_ratio"] = df.get("num_lab_procedures", 0) / (df.get("num_medications", 0) + 1.0)

    num_inp = df.get("number_inpatient", pd.Series(0, index=df.index))
    num_emg = df.get("number_emergency", pd.Series(0, index=df.index))
    num_out = df.get("number_outpatient", pd.Series(0, index=df.index))

    df["has_prior_inpatient"] = (num_inp > 0).astype(int)
    df["has_prior_emergency"] = (num_emg > 0).astype(int)
    df["has_prior_outpatient"] = (num_out > 0).astype(int)

    tot_v = df["total_prior_visits"].clip(lower=0) + 1.0
    df["inpatient_ratio"] = num_inp / tot_v
    df["emergency_ratio"] = num_emg / tot_v

    df["high_utilizer"] = (df["total_prior_visits"] >= 3).astype(int)
    df["polypharmacy"] = (df.get("num_medications", 0) >= 10).astype(int)
    df["clinical_complexity"] = (
        df.get("num_lab_procedures", 0)
        + (df.get("num_procedures", 0) * 2.5)
        + (df.get("number_diagnoses", 0) * 3.0)
    )

    df["inpatient_sq"] = num_inp ** 2
    df["emergency_sq"] = num_emg ** 2

    age_num = df.get("age_numeric", pd.Series(65.0, index=df.index))
    num_diag = df.get("number_diagnoses", pd.Series(0, index=df.index))
    df["age_inpatient_interaction"] = age_num * (num_inp + 1.0)
    df["age_diag_interaction"] = age_num * (num_diag + 1.0)

    active_meds = df.get("num_meds_active", pd.Series(0, index=df.index)).clip(lower=0) + 1.0
    df["med_change_ratio"] = df.get("num_med_changes", pd.Series(0, index=df.index)) / active_meds

    print(f"[clean] Output shape: {df.shape}")
    return df


def clean_inference_patient(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw patient DataFrame for production inference (single or batch).
    Robustly ensures all required feature columns exist with safe defaults.
    """
    df = df_raw.copy()
    df.replace("?", np.nan, inplace=True)

    # Defaults for optional / missing columns in raw input
    if "race" not in df.columns:
        df["race"] = "Unknown"
    else:
        df["race"] = df["race"].fillna("Unknown")

    if "gender" not in df.columns:
        df["gender"] = "Female"
    else:
        df["gender"] = df["gender"].fillna("Female")

    if "payer_code" not in df.columns:
        df["payer_code"] = "Unknown"
    else:
        df["payer_code"] = df["payer_code"].fillna("Unknown")

    if "medical_specialty" not in df.columns:
        df["medical_specialty"] = "Unknown"
    else:
        df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")

    for col in ["max_glu_serum", "A1Cresult"]:
        if col not in df.columns:
            df[col] = "Not_Tested"
        else:
            df[col] = df[col].fillna("Not_Tested")
            df[col] = df[col].replace({"nan": "Not_Tested", "None": "Not_Tested", "": "Not_Tested"})

    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if col not in df.columns:
            df[col] = "1"
        else:
            df[col] = df[col].apply(
                lambda x: str(int(x)) if pd.notnull(x) and isinstance(x, float) and x.is_integer()
                else str(x)
            )

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_group"] = df[col].apply(_map_icd9)
        elif f"{col}_group" not in df.columns:
            df[f"{col}_group"] = "Other"

    if "age" in df.columns:
        df["age_numeric"] = df["age"].apply(_age_midpoint)
    elif "age_numeric" not in df.columns:
        df["age_numeric"] = 65.0  # default adult age midpoint

    df["total_prior_visits"] = (
        df.get("number_outpatient", pd.Series(0, index=df.index))
        + df.get("number_emergency", pd.Series(0, index=df.index))
        + df.get("number_inpatient", pd.Series(0, index=df.index))
    )

    potential_med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
        "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
        "tolazamide", "examide", "citoglipton", "insulin",
        "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    med_cols = [c for c in potential_med_cols if c in df.columns]
    if med_cols:
        df["num_med_changes"] = (df[med_cols].isin(["Up", "Down"])).sum(axis=1)
        df["num_meds_active"] = (df[med_cols] != "No").sum(axis=1)
    else:
        if "num_med_changes" not in df.columns:
            df["num_med_changes"] = 0
        if "num_meds_active" not in df.columns:
            df["num_meds_active"] = 0

    # Clinical ratios, intensities & interaction features
    time_hosp = df.get("time_in_hospital", pd.Series(1, index=df.index)).clip(lower=1)
    df["labs_per_day"] = df.get("num_lab_procedures", 0) / time_hosp
    df["meds_per_day"] = df.get("num_medications", 0) / time_hosp
    df["procedures_per_day"] = df.get("num_procedures", 0) / time_hosp
    df["diagnoses_per_day"] = df.get("number_diagnoses", 0) / time_hosp
    df["lab_med_ratio"] = df.get("num_lab_procedures", 0) / (df.get("num_medications", 0) + 1.0)

    num_inp = df.get("number_inpatient", pd.Series(0, index=df.index))
    num_emg = df.get("number_emergency", pd.Series(0, index=df.index))
    num_out = df.get("number_outpatient", pd.Series(0, index=df.index))

    df["has_prior_inpatient"] = (num_inp > 0).astype(int)
    df["has_prior_emergency"] = (num_emg > 0).astype(int)
    df["has_prior_outpatient"] = (num_out > 0).astype(int)

    tot_v = df["total_prior_visits"].clip(lower=0) + 1.0
    df["inpatient_ratio"] = num_inp / tot_v
    df["emergency_ratio"] = num_emg / tot_v

    df["high_utilizer"] = (df["total_prior_visits"] >= 3).astype(int)
    df["polypharmacy"] = (df.get("num_medications", 0) >= 10).astype(int)
    df["clinical_complexity"] = (
        df.get("num_lab_procedures", 0)
        + (df.get("num_procedures", 0) * 2.5)
        + (df.get("number_diagnoses", 0) * 3.0)
    )

    df["inpatient_sq"] = num_inp ** 2
    df["emergency_sq"] = num_emg ** 2

    age_num = df.get("age_numeric", pd.Series(65.0, index=df.index))
    num_diag = df.get("number_diagnoses", pd.Series(0, index=df.index))
    df["age_inpatient_interaction"] = age_num * (num_inp + 1.0)
    df["age_diag_interaction"] = age_num * (num_diag + 1.0)

    active_meds = df.get("num_meds_active", pd.Series(0, index=df.index)).clip(lower=0) + 1.0
    df["med_change_ratio"] = df.get("num_med_changes", pd.Series(0, index=df.index)) / active_meds

    return df

