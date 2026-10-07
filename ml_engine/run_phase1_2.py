import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from ucimlrepo import fetch_ucirepo

from clean import clean_data
from split import split_data, build_preprocessors

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REPORTS_DIR = os.path.join(BASE_DIR, 'ml_engine', 'reports', 'eda')
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'ml_engine', 'data', 'processed')
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    print("Fetching dataset...")
    try:
        dataset = fetch_ucirepo(id=296)
        X = dataset.data.features
        y = dataset.data.targets
        ids = dataset.data.ids
        df = pd.concat([ids, X, y], axis=1)
    except Exception as e:
        print(f"Failed to fetch dataset via ucimlrepo: {e}")
        raw_path = os.path.join(BASE_DIR, 'ml_engine', 'data', 'raw', 'diabetic_data.csv')
        if os.path.exists(raw_path):
            print("Loading from local raw file.")
            df = pd.read_csv(raw_path)
        else:
            raise FileNotFoundError(f"Missing {raw_path} and download failed.")
            
    print(f"Loaded dataset: {df.shape}")
    
    print("\n--- PHASE 1: CLEANING ---")
    df_clean = clean_data(df)
    
    # Exclude identifiers from model features
    ident_cols = ['encounter_id', 'patient_nbr']
    
    # Determine numeric vs categorical
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ident_cols + ['readmitted_binary']]
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ident_cols]
    
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)
    
    # 1.5 EDA Plots
    print("Generating EDA plots...")
    
    # Class balance
    plt.figure()
    df_clean['readmitted_binary'].value_counts().plot(kind='bar', title='Class Balance')
    plt.savefig(os.path.join(REPORTS_DIR, 'class_balance.png'))
    plt.close()
    
    # Readmission by age numeric (wait, age_numeric was created)
    if 'age_numeric' in df_clean.columns:
        plt.figure()
        df_clean.groupby('age_numeric')['readmitted_binary'].mean().plot(kind='bar', title='Readmission Rate by Age')
        plt.savefig(os.path.join(REPORTS_DIR, 'readmission_by_age.png'))
        plt.close()
        
    # Readmission by diag_1_group
    if 'diag_1_group' in df_clean.columns:
        plt.figure()
        df_clean.groupby('diag_1_group')['readmitted_binary'].mean().plot(kind='bar', title='Readmission Rate by Diag 1')
        plt.savefig(os.path.join(REPORTS_DIR, 'readmission_by_diag1.png'))
        plt.close()
        
    # Readmission by number_inpatient bucket
    df_clean['inpatient_bucket'] = pd.cut(df_clean['number_inpatient'], bins=[-1, 0, 1, 3, 10, 100], labels=['0', '1', '2-3', '4-10', '>10'])
    plt.figure()
    df_clean.groupby('inpatient_bucket')['readmitted_binary'].mean().plot(kind='bar', title='Readmission Rate by Prior Inpatient')
    plt.savefig(os.path.join(REPORTS_DIR, 'readmission_by_inpatient.png'))
    plt.close()
    df_clean.drop(columns=['inpatient_bucket'], inplace=True)
    
    # Histogram grid
    df_clean[numeric_cols].hist(figsize=(15, 10), bins=20)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'numeric_histograms.png'))
    plt.close()
    
    print("\n--- PHASE 2: SPLIT & PREPROCESS ---")
    df_train, df_val, df_test = split_data(df_clean)
    
    # Assertions
    train_patients = set(df_train['patient_nbr'])
    val_patients = set(df_val['patient_nbr'])
    test_patients = set(df_test['patient_nbr'])
    
    assert train_patients.isdisjoint(val_patients), "Leakage: Train and Val share patients!"
    assert train_patients.isdisjoint(test_patients), "Leakage: Train and Test share patients!"
    assert val_patients.isdisjoint(test_patients), "Leakage: Val and Test share patients!"
    
    overall_pos = df_clean['readmitted_binary'].mean()
    print("Overall positive rate:", round(overall_pos, 4))
    
    for name, d in zip(['Train', 'Val', 'Test'], [df_train, df_val, df_test]):
        print(f"{name}: {len(d)} rows, {len(d)/len(df_clean):.1%} of data, positive rate {d['readmitted_binary'].mean():.4f}")
        
    # Build preprocessors
    prep_onehot, prep_ft = build_preprocessors(df_train, numeric_cols, categorical_cols)
    
    # Export for M2
    df_train.to_csv(os.path.join(DATA_PROCESSED_DIR, 'train_with_ids.csv'), index=False)
    df_val.to_csv(os.path.join(DATA_PROCESSED_DIR, 'val_with_ids.csv'), index=False)
    df_test.to_csv(os.path.join(DATA_PROCESSED_DIR, 'test_with_ids.csv'), index=False)
    
    split_manifest = pd.concat([
        df_train[['encounter_id', 'patient_nbr', 'split']],
        df_val[['encounter_id', 'patient_nbr', 'split']],
        df_test[['encounter_id', 'patient_nbr', 'split']]
    ])
    split_manifest.to_csv(os.path.join(DATA_PROCESSED_DIR, 'split_manifest.csv'), index=False)
    
    # Data Dictionary
    with open(os.path.join(DATA_PROCESSED_DIR, 'DATA_DICTIONARY.md'), 'w') as f:
        f.write("# Data Dictionary\n")
        f.write("Dataset cleaned. Missing values handled. Categorical features grouped (<0.5%).\n")
        f.write("Target: readmitted_binary (1 if <30 days, 0 otherwise).\n")
        f.write("Split: 80/10/10 stratified by target, grouped by patient_nbr.\n")
        
    with open(os.path.join(DATA_PROCESSED_DIR, 'RAG_HANDOFF.md'), 'w') as f:
        f.write("# RAG Feature Handoff\n")
        f.write("M2, please provide the RAG feature as one CSV per split at:\n")
        f.write("- `ml_engine/data/rag/rag_features_train.csv`\n")
        f.write("- `ml_engine/data/rag/rag_features_val.csv`\n")
        f.write("- `ml_engine/data/rag/rag_features_test.csv`\n\n")
        f.write("Columns required: `encounter_id`, `rag_neighborhood_readmit_rate`\n")
        f.write("The retrieval index must be built from TRAIN encounters ONLY.\n")
        f.write("CRITICAL: For TRAIN rows, retrieval MUST EXCLUDE the row itself AND all other encounters of the same `patient_nbr` (leave-one-out) to prevent leakage.\n")
        
    print("Phase 1 & 2 completed successfully. Preprocessors built and files exported.")
    
if __name__ == "__main__":
    main()
