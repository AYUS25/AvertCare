#!/usr/bin/env python3
"""
AvertCare · RAG Engine Seeding Pipeline
========================================
Ingests diabetic_data.csv → generates clinical discharge notes
→ embeds via sentence-transformers → upserts into Qdrant
→ computes RAGrisk for each patient → exports train_with_rag.csv for M1.

Usage:
    python scripts/seed_rag_engine.py \
        --csv data/raw/diabetic_data.csv \
        --qdrant http://localhost:6333 \
        --limit 10000

Requirements:
    pip install qdrant-client sentence-transformers pandas tqdm
"""

from __future__ import annotations

import argparse
import random
import uuid
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

COLLECTION_NAME = "clinical_cases"
MODEL_NAME = "all-MiniLM-L6-v2"       # 384-dim, fast, CPU-friendly
VECTOR_DIM = 384
BATCH_SIZE = 64
RAG_K = 5                              # top-k neighbors for RAGrisk

# SDoH clause pool — injected randomly into clinical notes
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

# Intervention pool — assigned to non-readmitted twin patients
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
# 1. Note Generator
# ─────────────────────────────────────────────────────────────

def generate_clinical_note(row: pd.Series) -> str:
    """
    Synthesise a realistic, structured clinical discharge summary
    from tabular row fields. Injects a random SDoH clause.
    """
    readmit_label = "has a history of inpatient readmission" if row.get("readmitted_binary", 0) == 1 \
        else "has no recent inpatient readmissions"

    medication_status = "medication regimen changed during this admission" \
        if str(row.get("change", "No")).lower() == "ch" \
        else "no significant medication changes during this admission"

    sdoh = random.choice(_SDOH_CLAUSES)

    note = (
        f"Patient is a {int(row.get('age_numeric', 65))}-year-old individual admitted for "
        f"{str(row.get('diag_1', 'unspecified diagnosis'))}. "
        f"Length of stay: {int(row.get('time_in_hospital', 5))} days. "
        f"Number of inpatient visits in the past year: {int(row.get('number_inpatient', 0))}. "
        f"Number of emergency visits: {int(row.get('number_emergency', 0))}. "
        f"Active medications at discharge: {int(row.get('num_medications', 1))}. "
        f"The patient {readmit_label}. "
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
# 3. Preprocessing helpers
# ─────────────────────────────────────────────────────────────

AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Readmission binary flag
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

    # Numeric age
    df["age_numeric"] = df["age"].map(AGE_MAP).fillna(65).astype(int)

    # Stable patient ID string
    if "patient_nbr" not in df.columns:
        df["patient_id"] = [f"SYN-{i:06d}" for i in range(len(df))]
    else:
        df["patient_id"] = df["patient_nbr"].astype(str)

    return df


# ─────────────────────────────────────────────────────────────
# 4. Main Pipeline
# ─────────────────────────────────────────────────────────────

def run(csv_path: Path, qdrant_url: str, limit: int) -> None:
    print("\n=== AvertCare · RAG Seeding Pipeline ===\n")

    # Load & preprocess
    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path, nrows=limit if limit > 0 else None)
    df = preprocess(df)
    print(f"   {len(df):,} patient records loaded.\n")

    # Generate clinical notes
    print("📝 Generating clinical notes…")
    df["clinical_note"] = df.apply(generate_clinical_note, axis=1)

    # Load embedding model
    print(f"🤖 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Batch embed
    print(f"🔢 Embedding {len(df):,} notes in batches of {BATCH_SIZE}…")
    all_notes = df["clinical_note"].tolist()
    embeddings = model.encode(
        all_notes,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,     # cosine similarity → dot product
    )

    # Connect to Qdrant
    print(f"\n📡 Connecting to Qdrant at {qdrant_url}…")
    client = QdrantClient(url=qdrant_url, timeout=60)
    initialise_collection(client)

    # Upsert in batches
    print("⬆️  Upserting vectors…")
    points: list[PointStruct] = []

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        readmitted = int(row["readmitted_binary"])
        # Only non-readmitted patients get interventions
        interventions = (
            random.sample(_INTERVENTIONS_POOL, k=random.randint(2, 4))
            if readmitted == 0
            else []
        )

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[idx].tolist(),
                payload={
                    "patient_id": str(row["patient_id"]),
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
    print(f"\n✅ Upserted {total:,} vectors into '{COLLECTION_NAME}'.\n")

    # RAGrisk computation
    print(f"🔬 Computing RAGrisk (top-{RAG_K} neighbors) for each patient…")
    rag_risks: list[float] = []

    for idx in tqdm(range(len(df))):
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embeddings[idx].tolist(),
            limit=RAG_K + 1,           # +1 because the point matches itself
            with_payload=True,
        )
        results = response.points
        # Exclude self-match (score ≈ 1.0)
        neighbors = [r for r in results if r.score < 0.9999][:RAG_K]

        if neighbors:
            readmit_count = sum(r.payload.get("readmitted_binary", 0) for r in neighbors)
            rag_risk = readmit_count / len(neighbors)
        else:
            rag_risk = 0.0

        rag_risks.append(round(rag_risk, 4))

    df["rag_readmit_rate"] = rag_risks

    # Export for M1
    out_path = Path("data/processed/train_with_rag.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n📦 M1 training dataset exported → {out_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns added: 'clinical_note', 'age_numeric', 'readmitted_binary', 'rag_readmit_rate'\n")
    print("=== Pipeline Complete ✅ ===\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AvertCare RAG Seeding Pipeline")
    parser.add_argument("--csv", type=Path, default=Path("data/raw/diabetic_data.csv"))
    parser.add_argument("--qdrant", type=str, default="http://localhost:6333")
    parser.add_argument("--limit", type=int, default=10_000, help="Row limit (0 = all)")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {args.csv}\n"
            "   Download from: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008\n"
            "   Place as: data/raw/diabetic_data.csv\n"
        )

    run(args.csv, args.qdrant, args.limit)
