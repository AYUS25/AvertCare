#!/usr/bin/env python3
"""
Fast streaming seeder for Qdrant using precomputed training embeddings.
Seeds the collection `clinical_cases` in local Qdrant.
"""

import time
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "clinical_cases"
INDEX_PATH = "/app/models/rag_train_index.npz"
META_PATH = "/app/models/rag_train_meta.csv"
BATCH_SIZE = 2500

def seed():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    
    print(f"Loading precomputed index from {INDEX_PATH} and {META_PATH}...")
    loaded = np.load(INDEX_PATH)
    embeddings = loaded["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / np.clip(norms, 1e-8, None)).astype(np.float32)
    
    meta = pd.read_csv(META_PATH)
    total_points = len(embeddings)
    print(f"Loaded {total_points} vectors and metadata rows.")
    
    # Recreate collection
    print(f"Recreating collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    
    print(f"Streaming {total_points} points in batches of {BATCH_SIZE}...")
    start_time = time.time()
    
    for start_idx in range(0, total_points, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_points)
        batch_vectors = embeddings[start_idx:end_idx]
        batch_meta = meta.iloc[start_idx:end_idx]
        
        points = []
        for i, (vec, (_, row)) in enumerate(zip(batch_vectors, batch_meta.iterrows())):
            point_id = int(start_idx + i)
            payload = {
                "encounter_id": int(row["encounter_id"]),
                "patient_id": str(row["patient_id"]),
                "age": float(row["age"]),
                "primary_diagnosis": str(row["primary_diagnosis"]),
                "readmitted_binary": int(row["readmitted_binary"]),
                "number_inpatient": int(row.get("number_inpatient", 0)),
                "time_in_hospital": int(row.get("time_in_hospital", 0)),
                "interventions": str(row.get("interventions", "")),
            }
            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        elapsed = time.time() - start_time
        pct = (end_idx / total_points) * 100
        print(f"  Upserted {end_idx}/{total_points} ({pct:.1f}%) in {elapsed:.1f}s")
    
    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"✅ Seeding complete! Total points in Qdrant '{COLLECTION_NAME}': {count}")

if __name__ == "__main__":
    seed()
