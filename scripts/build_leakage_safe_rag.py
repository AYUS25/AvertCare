#!/usr/bin/env python3
"""Build a leakage-safe Twin-Patient RAG feature.

Does not modify data/processed/train_with_rag.csv and does not touch Qdrant.

Rules enforced here:
  1. Notes use discharge-time fields only. No readmission language.
  2. The neighbor index contains training encounters only.
  3. A training query excludes every encounter of the same patient_nbr.
  4. Validation and test queries retrieve from the training index only.
     Patient-grouped splits already keep those patients out of train.

Writes:
  data/processed/rag_features_safe.csv
  backend/models/rag_train_index.npz
  backend/models/rag_train_meta.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_engine.note_template import build_discharge_note, historical_interventions  # noqa: E402

DATA_DIR = ROOT / "ml_engine" / "data" / "processed"
OUT_CSV = ROOT / "data" / "processed" / "rag_features_safe.csv"
INDEX_NPZ = ROOT / "backend" / "models" / "rag_train_index.npz"
META_CSV = ROOT / "backend" / "models" / "rag_train_meta.csv"
CACHE_DIR = ROOT / "data" / "processed" / "rag_safe_cache"
KS = (5, 10, 20)
BATCH = 512


def _load_splits() -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(DATA_DIR / f"{name}_with_ids.csv")
        df["split"] = name
        df["patient_nbr"] = df["patient_nbr"].astype(np.int64)
        df["encounter_id"] = df["encounter_id"].astype(np.int64)
        splits[name] = df
    train_pts = set(splits["train"]["patient_nbr"])
    val_pts = set(splits["val"]["patient_nbr"])
    test_pts = set(splits["test"]["patient_nbr"])
    if train_pts & val_pts or train_pts & test_pts or val_pts & test_pts:
        raise RuntimeError("Patient overlap across splits. Refusing to build RAG features.")
    return splits


def _embed(notes: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(
        notes,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def _patient_index(patients: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(patients, kind="mergesort")
    sorted_pts = patients[order]
    change = np.flatnonzero(np.diff(sorted_pts)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(sorted_pts)]
    return {int(sorted_pts[s]): order[s:e] for s, e in zip(starts, ends)}


def _neighborhood_rates(
    query_emb: np.ndarray,
    query_patients: np.ndarray,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    patient_to_idx: dict[int, np.ndarray],
    exclude_same_patient: bool,
) -> dict[int, np.ndarray]:
    n_query = query_emb.shape[0]
    take = max(KS)
    out = {k: np.empty(n_query, dtype=np.float32) for k in KS}
    for start in range(0, n_query, BATCH):
        end = min(start + BATCH, n_query)
        sims = query_emb[start:end] @ train_emb.T
        for local_i, global_i in enumerate(range(start, end)):
            scores = sims[local_i]
            if exclude_same_patient:
                banned = patient_to_idx.get(int(query_patients[global_i]))
                if banned is not None and len(banned):
                    scores = scores.copy()
                    scores[banned] = -1.0
            part = np.argpartition(-scores, take - 1)[:take]
            top = part[np.argsort(-scores[part])]
            labels = train_labels[top]
            for k in KS:
                out[k][global_i] = float(labels[:k].mean())
        if end == n_query or (start // BATCH) % 20 == 0:
            print(f"    neighbors {end:,}/{n_query:,}", flush=True)
    return out


def _audit(frame: pd.DataFrame, train_patients: set[int]) -> None:
    notes = frame["clinical_note"].str.lower()
    leaked = notes.str.contains("readmit", regex=False) | notes.str.contains("readmission", regex=False)
    if int(leaked.sum()):
        raise RuntimeError(f"{int(leaked.sum())} notes still contain an outcome phrase.")

    for split in ("train", "val", "test"):
        sub = frame[frame["split"] == split]
        corr = sub["rag_readmit_rate_k10"].corr(sub["readmitted_binary"])
        print(
            f"  {split:5s} n={len(sub):6d}  "
            f"mean_k10={sub['rag_readmit_rate_k10'].mean():.4f}  "
            f"corr_k10={corr:.4f}  "
            f"mean_k10|y1={sub.loc[sub.readmitted_binary==1, 'rag_readmit_rate_k10'].mean():.4f}  "
            f"mean_k10|y0={sub.loc[sub.readmitted_binary==0, 'rag_readmit_rate_k10'].mean():.4f}"
        )
        if corr > 0.7:
            raise RuntimeError(f"{split} RAG correlation {corr:.3f} is high enough to indicate leakage.")

    heldout_pts = set(frame.loc[frame["split"] != "train", "patient_nbr"].astype(np.int64))
    if heldout_pts & train_patients:
        raise RuntimeError("Held-out patients are present in the training patient set.")
    print("  Audit passed: no outcome text, correlation below 0.70, no held-out patients in the index.")


def main() -> None:
    print("Loading patient-grouped splits...")
    splits = _load_splits()
    train = splits["train"].reset_index(drop=True)
    val = splits["val"].reset_index(drop=True)
    test = splits["test"].reset_index(drop=True)
    print(f"  train {len(train):,}  val {len(val):,}  test {len(test):,}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = CACHE_DIR / "embeddings_float32.npy"
    notes_path = CACHE_DIR / "notes.csv"

    frames = {"train": train, "val": val, "test": test}
    if emb_path.exists() and notes_path.exists():
        print(f"Reusing cached embeddings at {emb_path}")
        all_notes_df = pd.read_csv(notes_path)
        embeddings = np.load(emb_path)
    else:
        print("Writing discharge-time notes...")
        pieces = []
        for name, df in frames.items():
            notes = [build_discharge_note(rec) for rec in df.to_dict(orient="records")]
            piece = df[["encounter_id", "patient_nbr", "readmitted_binary"]].copy()
            piece["split"] = name
            piece["clinical_note"] = notes
            pieces.append(piece)
        all_notes_df = pd.concat(pieces, ignore_index=True)
        print(f"Embedding {len(all_notes_df):,} notes with all-MiniLM-L6-v2...")
        embeddings = _embed(all_notes_df["clinical_note"].tolist())
        np.save(emb_path, embeddings)
        all_notes_df.to_csv(notes_path, index=False)
        print(f"  cached {emb_path}")

    if len(all_notes_df) != len(embeddings):
        raise RuntimeError("Embedding cache does not match the note table.")

    n_train = len(train)
    train_emb = embeddings[:n_train]
    val_emb = embeddings[n_train:n_train + len(val)]
    test_emb = embeddings[n_train + len(val):]
    train_labels = train["readmitted_binary"].to_numpy(dtype=np.float32)
    patient_to_idx = _patient_index(train["patient_nbr"].to_numpy())

    print("Scoring training queries (same-patient encounters excluded)...")
    train_rates = _neighborhood_rates(
        train_emb, train["patient_nbr"].to_numpy(), train_emb, train_labels, patient_to_idx, True
    )
    print("Scoring validation queries against the training index...")
    val_rates = _neighborhood_rates(
        val_emb, val["patient_nbr"].to_numpy(), train_emb, train_labels, patient_to_idx, False
    )
    print("Scoring test queries against the training index...")
    test_rates = _neighborhood_rates(
        test_emb, test["patient_nbr"].to_numpy(), train_emb, train_labels, patient_to_idx, False
    )

    pieces = []
    for df, rates in ((train, train_rates), (val, val_rates), (test, test_rates)):
        out = df[["encounter_id", "patient_nbr", "readmitted_binary", "split"]].copy()
        for k, values in rates.items():
            out[f"rag_readmit_rate_k{k}"] = values
        pieces.append(out)
    feature_frame = pd.concat(pieces, ignore_index=True)

    # Attach notes for the audit, then drop them from the modeling CSV.
    feature_frame = feature_frame.merge(
        all_notes_df[["encounter_id", "clinical_note"]], on="encounter_id", how="left"
    )
    print("Audit:")
    _audit(feature_frame, set(train["patient_nbr"].tolist()))

    keep = [
        "encounter_id",
        "patient_nbr",
        "split",
        "rag_readmit_rate_k5",
        "rag_readmit_rate_k10",
        "rag_readmit_rate_k20",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feature_frame[keep].to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(feature_frame):,} rows). Existing train_with_rag.csv was not modified.")

    meta = train[
        [
            "encounter_id",
            "patient_nbr",
            "age_numeric",
            "diag_1_group",
            "readmitted_binary",
            "number_inpatient",
            "time_in_hospital",
            "num_medications",
            "num_med_changes",
            "number_emergency",
            "A1Cresult",
            "diabetesMed",
            "payer_code",
        ]
    ].copy()
    meta.insert(0, "row_idx", np.arange(len(meta), dtype=np.int32))
    meta["interventions"] = [
        " | ".join(historical_interventions(rec) if int(rec["readmitted_binary"]) == 0 else [])
        for rec in meta.to_dict(orient="records")
    ]
    meta = meta.rename(columns={"patient_nbr": "patient_id", "diag_1_group": "primary_diagnosis", "age_numeric": "age"})
    meta.to_csv(META_CSV, index=False)
    np.savez_compressed(
        INDEX_NPZ,
        embeddings=train_emb.astype(np.float16),
        labels=train_labels.astype(np.uint8),
        ks=np.array(KS, dtype=np.int16),
    )
    print(f"Wrote serving index {INDEX_NPZ} and {META_CSV}")
    print("Qdrant collection clinical_cases was not modified.")


if __name__ == "__main__":
    main()
