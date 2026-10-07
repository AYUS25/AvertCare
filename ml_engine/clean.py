import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    print(f"Original shape: {df.shape}")
    
    # 1.2 Target
    if 'readmitted' in df.columns:
        df['readmitted_binary'] = (df['readmitted'] == "<30").astype(int)
        df.drop(columns=['readmitted'], inplace=True)
        print("Class balance (readmitted_binary):")
        print(df['readmitted_binary'].value_counts(normalize=True))
    
    # 1.3a Replace "?" with missing
    df.replace("?", np.nan, inplace=True)
    
    # 1.3b Drop `weight`
    if 'weight' in df.columns:
        df.drop(columns=['weight'], inplace=True)
        
    if 'payer_code' in df.columns:
        df['payer_code'] = df['payer_code'].fillna("Unknown")
    if 'medical_specialty' in df.columns:
        df['medical_specialty'] = df['medical_specialty'].fillna("Unknown")
    
        # group medical_specialty categories with < 1%
        spec_counts = df['medical_specialty'].value_counts(normalize=True)
        valid_specs = spec_counts[spec_counts >= 0.01].index
        df['medical_specialty'] = df['medical_specialty'].where(df['medical_specialty'].isin(valid_specs), "Other")
    
    if 'race' in df.columns:
        df['race'] = df['race'].fillna("Unknown")
    
    # 1.3c Remove rows with gender "Unknown/Invalid"
    if 'gender' in df.columns:
        df = df[df['gender'] != "Unknown/Invalid"]
    
    # 1.3d Remove death/hospice (11, 13, 14, 19, 20, 21)
    if 'discharge_disposition_id' in df.columns:
        df['discharge_disposition_id'] = pd.to_numeric(df['discharge_disposition_id'], errors='coerce')
        death_hospice_ids = [11, 13, 14, 19, 20, 21]
        mask = df['discharge_disposition_id'].isin(death_hospice_ids)
        print(f"Removing {mask.sum()} encounters due to death/hospice discharge.")
        df = df[~mask]
    
    # 1.3e Drop constant/near-constant columns
    dropped_const = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            dropped_const.append(col)
            df.drop(columns=[col], inplace=True)
    if dropped_const:
        print(f"Dropped constant columns: {dropped_const}")
            
    # 1.3f Convert to categorical strings, group < 0.5%
    for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        if col in df.columns:
            # First, handle numeric floats ending in .0 if any
            df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) and isinstance(x, float) and x.is_integer() else str(x))
            counts = df[col].value_counts(normalize=True)
            valid = counts[counts >= 0.005].index
            df[col] = df[col].where(df[col].isin(valid), "Other")
            
    # 1.3g Group ICD-9 codes
    def map_icd9(val):
        if pd.isna(val) or val == "Unknown" or val == "nan":
            return "Other"
        val_str = str(val).strip()
        if val_str.startswith("250"):
            return "Diabetes"
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
            
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[f"{col}_group"] = df[col].apply(map_icd9)
            df.drop(columns=[col], inplace=True)
            
    # 1.3h age midpoint
    def age_midpoint(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip("[]()")
        parts = val.split("-")
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        return np.nan
    
    if 'age' in df.columns:
        df['age_numeric'] = df['age'].apply(age_midpoint)
        df.drop(columns=['age'], inplace=True)
        
    # 1.3i max_glu_serum and A1Cresult
    for col in ['max_glu_serum', 'A1Cresult']:
        if col in df.columns:
            df[col] = df[col].fillna("None")
            df[col] = df[col].replace("nan", "None") # Just in case it became a string
            
    # 1.4 Engineered features
    df['total_prior_visits'] = df.get('number_outpatient', 0) + df.get('number_emergency', 0) + df.get('number_inpatient', 0)
    
    # Medication columns have categories No, Steady, Up, Down. Find them dynamically:
    # Based on UCI dataset knowledge, these are the typical med columns
    potential_med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                          'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 
                          'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 
                          'tolazamide', 'examide', 'citoglipton', 'insulin', 
                          'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                          'metformin-rosiglitazone', 'metformin-pioglitazone']
    
    med_cols = [c for c in potential_med_cols if c in df.columns]
            
    if med_cols:
        df['num_med_changes'] = (df[med_cols].isin(["Up", "Down"])).sum(axis=1)
        df['num_meds_active'] = (df[med_cols] != "No").sum(axis=1)
        
        dropped_meds = []
        for col in med_cols:
            pct_no = (df[col] == "No").mean()
            if pct_no > 0.99:
                dropped_meds.append(col)
                df.drop(columns=[col], inplace=True)
        if dropped_meds:
            print(f"Dropped medication columns (>99% 'No'): {dropped_meds}")
                
    return df
