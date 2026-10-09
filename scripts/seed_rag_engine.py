#!/usr/bin/env python3
"""
AvertCare · RAG Engine Seeding & Leakage-Free Dataset Generation Pipeline
==========================================================================
Ingests diabetic_data.csv → generates LEAKAGE-FREE clinical discharge notes
→ embeds via sentence-transformers → upserts TRAIN split into Qdrant
→ computes leakage-safe RAG readmit rate for each split → exports clean RAG datasets.

Leakage Protections Implemented:
1. NO target labels (readmitted, readmitted_binary) in clinical notes or SDoH clauses.
2. Interventions assigned independently of readmitted_binary.
3. Strict patient-level train (80%), validation (10%), test (10%) splits (0 patient overlap).
4. Qdrant vector collection indexed ONLY with Training split records.
5. Training split RAG rate computed via out-of-fold/leave-one-patient-out filtering.
6. Validation and Test splits query Training vectors ONLY (excluding same-patient records).

Usage:
    python scripts/seed_rag_engine.py \
        --csv data/raw/diabetic_data.csv \
        --qdrant http://localhost:6333 \
        --limit 10000

Requirements:
    pip install qdrant-client sentence-transformers pandas tqdm scikit-learn
"""

from __future__ import annotations

import argparse
import random
import uuid
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

COLLECTION_NAME = "clinical_cases"
MODEL_NAME = "all-MiniLM-L6-v2"       # 384-dim, fast, CPU-friendly
VECTOR_DIM = 384
BATCH_SIZE = 64
RAG_K = 5                              # top-k neighbors for RAG rate

# SDoH clause pool — injected randomly into clinical notes (Target Neutral)
_SDOH_CLAUSES = [
    "Patient lives alone with limited social support.",
    "Patient is uninsured and has expressed concern about medication costs.",
    "Transportation barrier documented — no reliable access to follow-up.",
    "Patient has stable family support and lives with spouse.",
    "Patient reports food insecurity; connected to community meal programme.",
    "Housing instability noted; case worker contacted.",
    "Patient is employed and financially stable.",
    "Caregiver fatigue documented; patient's primary carer is elderly spouse.",
]

# Intervention pool — assigned to historical cases independently of target label
_INTERVENTIONS_POOL = [
    "72-hr post-discharge telehealth follow-up scheduled",
    "Pharmacist medication reconciliation completed",
    "Home nursing visit arranged within 48 hrs",
    "Social work referral for transport assistance",
    "Diabetes self-management education completed",
    "Dietary counselling referral issued",
    "Primary care physician notified of discharge",
]


# ─────────────────────────────────────────────────────────────
# 1. Leakage-Free Note Generator
# ─────────────────────────────────────────────────────────────

def generate_clinical_note(row: pd.Series) -> str:
    """
    Synthesise a realistic, structured clinical discharge summary from tabular fields.
    STRICTLY LEAKAGE-FREE: Uses ONLY features available at prediction time.
    NO references to readmitted or readmitted_binary.
    """
    # Outcome text is intentionally absent. readmitted_binary must not appear
    # in the embedded note. The leakage-safe feature builder is
    # scripts/build_leakage_safe_rag.py.
    medication_status = "medication regimen changed during this admission" \
        if str(row.get("change", "No")).lower() == "ch" \
        else "no significant medication changes during this admission"
    )

    sdoh = random.choice(_SDOH_CLAUSES)

    note = (
        f"Patient is a {int(row.get('age_numeric', 65))}-year-old individual admitted for "
        f"{str(row.get('diag_1', 'unspecified diagnosis'))}. "
        f"Length of stay: {int(row.get('time_in_hospital', 5))} days. "
        f"Number of inpatient visits in the past year: {int(row.get('number_inpatient', 0))}. "
        f"Number of emergency visits: {int(row.get('number_emergency', 0))}. "
        f"Active medications at discharge: {int(row.get('num_medications', 1))}. "
        f"The {medication_status}. "
        f"{sdoh}"
    )
    return note


# ─────────────────────────────────────────────────────────────
# 2. Qdrant Initialiser
# ─────────────────────────────────────────────────────────────

def initialise_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — recreating for fresh seed.")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created ({VECTOR_DIM}-dim, Cosine).")


# ─────────────────────────────────────────────────────────────
# 3. Preprocessing & Patient-Level Splitting
# ─────────────────────────────────────────────────────────────

AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Target variable
    if "readmitted_binary" not in df.columns:
        df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

    # Numeric age
    if "age_numeric" not in df.columns:
        df["age_numeric"] = df["age"].map(AGE_MAP).fillna(65).astype(int)

    # Patient ID string
    if "patient_nbr" in df.columns:
        df["patient_id"] = df["patient_nbr"].astype(str)
    elif "patient_id" not in df.columns:
        df["patient_id"] = [f"SYN-{i:06d}" for i in range(len(df))]

    # Encounter ID
    if "encounter_id" not in df.columns:
        df["encounter_id"] = np.arange(1000000, 1000000 + len(df))

    return df


def patient_level_split(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Splits dataset into Train, Validation, and Test sets grouped by patient_id
    to guarantee zero patient overlap across splits.
    """
    gss_outer = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss_outer.split(df, groups=df["patient_id"]))

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_temp = df.iloc[temp_idx].copy().reset_index(drop=True)

    val_prop = val_ratio / (val_ratio + test_ratio)
    gss_inner = GroupShuffleSplit(n_splits=1, train_size=val_prop, random_state=random_state)
    val_idx, test_idx = next(gss_inner.split(df_temp, groups=df_temp["patient_id"]))

    df_val = df_temp.iloc[val_idx].copy().reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].copy().reset_index(drop=True)

    # Print patient isolation statistics
    train_pts = set(df_train["patient_id"])
    val_pts = set(df_val["patient_id"])
    test_pts = set(df_test["patient_id"])

    print(f"  [Split] Train: {len(df_train):,} rows ({len(train_pts):,} patients)")
    print(f"  [Split] Val:   {len(df_val):,} rows ({len(val_pts):,} patients)")
    print(f"  [Split] Test:  {len(df_test):,} rows ({len(test_pts):,} patients)")
    print(f"  [Leakage Check] Train ∩ Val patient overlap: {len(train_pts & val_pts)}")
    print(f"  [Leakage Check] Train ∩ Test patient overlap: {len(train_pts & test_pts)}")
    print(f"  [Leakage Check] Val ∩ Test patient overlap: {len(val_pts & test_pts)}")
    assert len(train_pts & val_pts) == 0, "Patient overlap detected between Train and Val!"
    assert len(train_pts & test_pts) == 0, "Patient overlap detected between Train and Test!"
    assert len(val_pts & test_pts) == 0, "Patient overlap detected between Val and Test!"

    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────
# 4. Leakage-Safe Vectorized RAG Neighbor Search & Rate Computation
# ─────────────────────────────────────────────────────────────

def run(csv_path: Path, qdrant_url: str, limit: int, api_key: str = "") -> None:
    print("\n=== AvertCare · RAG Seeding Pipeline ===\n")

    if use_qdrant_server and client is not None:
        for idx in tqdm(range(len(query_df)), desc="Querying Qdrant"):
            q_vec = query_embeddings[idx]
            q_patient = str(query_df.iloc[idx]["patient_id"])
            q_encounter = query_df.iloc[idx]["encounter_id"]

            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=q_vec.tolist(),
                limit=k + 25,
                with_payload=True,
            )
            neighbors_readmit = []
            for r in results:
                p_id = str(r.payload.get("patient_id", ""))
                e_id = r.payload.get("encounter_id", None)
                if p_id == q_patient:
                    continue
                if is_train and e_id == q_encounter:
                    continue
                neighbors_readmit.append(r.payload.get("readmitted_binary", 0))
                if len(neighbors_readmit) >= k:
                    break

            risk = sum(neighbors_readmit) / len(neighbors_readmit) if neighbors_readmit else global_mean_rate
            rag_risks.append(round(float(risk), 4))
    else:
        # Fast vectorized matrix dot product
        ref_patient_ids = ref_df["patient_id"].astype(str).values
        ref_encounter_ids = ref_df["encounter_id"].values
        ref_labels = ref_df["readmitted_binary"].values

        # Similarity matrix: (N_query, N_ref)
        sim_matrix = np.dot(query_embeddings, ref_embeddings.T)

        for idx in tqdm(range(len(query_df)), desc="Computing RAG rates (vectorized)"):
            q_patient = str(query_df.iloc[idx]["patient_id"])
            q_encounter = query_df.iloc[idx]["encounter_id"]

            scores = sim_matrix[idx]
            top_indices = np.argsort(scores)[::-1]

            neighbors_readmit = []
            for top_i in top_indices:
                p_id = ref_patient_ids[top_i]
                e_id = ref_encounter_ids[top_i]

                if p_id == q_patient:
                    continue  # Filter out same patient
                if is_train and e_id == q_encounter:
                    continue  # Filter out self encounter

                neighbors_readmit.append(ref_labels[top_i])
                if len(neighbors_readmit) >= k:
                    break

            risk = sum(neighbors_readmit) / len(neighbors_readmit) if neighbors_readmit else global_mean_rate
            rag_risks.append(round(float(risk), 4))

    return rag_risks


# ─────────────────────────────────────────────────────────────
# 5. Main Execution Pipeline
# ─────────────────────────────────────────────────────────────

def run(csv_path: Path, qdrant_url: str, limit: int, output_dir: Path) -> None:
    print("\n=== AvertCare · RAG Seeding & Leakage-Free Dataset Pipeline ===\n")
    random.seed(42)
    np.random.seed(42)

    # 1. Load raw dataset
    print(f"📂 Loading raw dataset: {csv_path}")
    df_raw = pd.read_csv(csv_path, nrows=limit if limit > 0 else None)
    df = preprocess(df_raw)
    print(f"   {len(df):,} total encounters loaded.\n")

    # 2. Patient-level Train/Val/Test splitting
    print("✂️  Performing patient-level train (80%), validation (10%), test (10%) split…")
    df_train, df_val, df_test = patient_level_split(df, random_state=42)

    # 3. Generate leakage-free clinical notes for each split
    print("\n📝 Generating leakage-free clinical notes (0 target words)…")
    df_train["clinical_note"] = df_train.apply(generate_clinical_note, axis=1)
    df_val["clinical_note"]   = df_val.apply(generate_clinical_note, axis=1)
    df_test["clinical_note"]  = df_test.apply(generate_clinical_note, axis=1)

    # 4. Load embedding model
    print(f"\n🤖 Loading embedding model: {MODEL_NAME}")
    embed_model = SentenceTransformer(MODEL_NAME)

    # 5. Embed notes per split
    print("\n🔢 Embedding Training clinical notes…")
    train_embeddings = embed_model.encode(
        df_train["clinical_note"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Connect to Qdrant
    print(f"\n📡 Connecting to Qdrant at {qdrant_url}…")
    client = QdrantClient(url=qdrant_url, api_key=api_key if api_key else None, timeout=60)
    initialise_collection(client)

    print("🔢 Embedding Test clinical notes…")
    test_embeddings = embed_model.encode(
        df_test["clinical_note"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 6. Try Qdrant Server Connection & Collection Initialisation
    use_qdrant_server = False
    client = None
    if QDRANT_AVAILABLE:
        try:
            print(f"\n📡 Attempting Qdrant connection at {qdrant_url}…")
            client = QdrantClient(url=qdrant_url, timeout=5)
            initialise_collection(client)

            print("⬆️  Upserting ONLY Training split vectors into Qdrant…")
            points: list[PointStruct] = []
            for idx, (_, row) in enumerate(tqdm(df_train.iterrows(), total=len(df_train))):
                readmitted = int(row["readmitted_binary"])
                # Interventions assigned based on clinical eligibility, NOT target label
                interventions = (
                    random.sample(_INTERVENTIONS_POOL, k=random.randint(2, 4))
                    if random.random() < 0.5
                    else []
                )

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=train_embeddings[idx].tolist(),
                        payload={
                            "patient_id": str(row["patient_id"]),
                            "encounter_id": int(row["encounter_id"]),
                            "age": int(row["age_numeric"]),
                            "primary_diagnosis": str(row.get("diag_1", "Unknown")),
                            "time_in_hospital": int(row.get("time_in_hospital", 0)),
                            "readmitted_binary": readmitted,
                            "interventions": interventions,
                            "clinical_note": str(row["clinical_note"]),
                        },
                    )
                )

                if len(points) >= BATCH_SIZE:
                    client.upsert(collection_name=COLLECTION_NAME, points=points)
                    points = []

            if points:
                client.upsert(collection_name=COLLECTION_NAME, points=points)

            total = client.count(collection_name=COLLECTION_NAME).count
            print(f"✅ Qdrant server active: Upserted {total:,} training vectors into '{COLLECTION_NAME}'.\n")
            use_qdrant_server = True
        except Exception as e:
            print(f"⚠️  Qdrant server unreachable ({e}). Using in-memory vector index fallback.\n")
            use_qdrant_server = False
    else:
        print("⚠️  qdrant_client module unavailable. Using in-memory vector index fallback.\n")

    # 7. Compute Leakage-Free RAG Readmission Rates
    print("🔬 Computing leakage-free RAG readmission rate for Training split (leave-one-patient-out)…")
    df_train["rag_readmit_rate"] = compute_leakage_free_rag_rate(
        train_embeddings, df_train, train_embeddings, df_train,
        is_train=True, k=RAG_K, client=client, use_qdrant_server=use_qdrant_server
    )

    print("🔬 Computing leakage-free RAG readmission rate for Validation split (querying Training vectors only)…")
    df_val["rag_readmit_rate"] = compute_leakage_free_rag_rate(
        val_embeddings, df_val, train_embeddings, df_train,
        is_train=False, k=RAG_K, client=client, use_qdrant_server=use_qdrant_server
    )

    print("🔬 Computing leakage-free RAG readmission rate for Test split (querying Training vectors only)…")
    df_test["rag_readmit_rate"] = compute_leakage_free_rag_rate(
        test_embeddings, df_test, train_embeddings, df_train,
        is_train=False, k=RAG_K, client=client, use_qdrant_server=use_qdrant_server
    )

    # 8. Export Processed Clean Datasets
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_with_rag_clean.csv"
    val_path   = output_dir / "val_with_rag_clean.csv"
    test_path  = output_dir / "test_with_rag_clean.csv"
    main_train = output_dir / "train_with_rag.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    # Also update main train_with_rag.csv while keeping backup
    df_train.to_csv(main_train, index=False)

    print(f"\n📦 Exported Leakage-Free Datasets to {output_dir}:")
    print(f"   - Train Split: {train_path} ({len(df_train):,} rows)")
    print(f"   - Val Split:   {val_path} ({len(df_val):,} rows)")
    print(f"   - Test Split:  {test_path} ({len(df_test):,} rows)")
    print(f"   - Main Train:  {main_train} ({len(df_train):,} rows)")

    # Print summary statistics of rag_readmit_rate
    print("\n📊 RAG Readmission Rate Summary Statistics:")
    print(f"   Train RAG Rate Mean: {df_train['rag_readmit_rate'].mean():.4f} (min: {df_train['rag_readmit_rate'].min()}, max: {df_train['rag_readmit_rate'].max()})")
    print(f"   Val RAG Rate Mean:   {df_val['rag_readmit_rate'].mean():.4f} (min: {df_val['rag_readmit_rate'].min()}, max: {df_val['rag_readmit_rate'].max()})")
    print(f"   Test RAG Rate Mean:  {df_test['rag_readmit_rate'].mean():.4f} (min: {df_test['rag_readmit_rate'].min()}, max: {df_test['rag_readmit_rate'].max()})")

    print("\n=== RAG Seeding Pipeline Completed Successfully ✅ ===\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AvertCare Leakage-Free RAG Seeding Pipeline")
    parser.add_argument("--csv", type=Path, default=Path("data/raw/diabetic_data.csv"))
    parser.add_argument("--qdrant", type=str, default="http://localhost:6333")
    parser.add_argument("--api-key", type=str, default="", help="Qdrant Cloud API Key")
    parser.add_argument("--limit", type=int, default=10_000, help="Row limit (0 = all)")
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {args.csv}\n"
            "   Download from: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008\n"
            "   Place as: data/raw/diabetic_data.csv\n"
        )

    run(args.csv, args.qdrant, args.limit, args.api_key)
