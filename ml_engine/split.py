import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

def split_data(df: pd.DataFrame, test_size=0.1, val_size=0.1, random_state=42):
    # We want 80/10/10 split
    # Since StratifiedGroupKFold splits into k folds, we can use k=10.
    # 1 fold for test, 1 for val, 8 for train.
    
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    groups = df['patient_nbr']
    y = df['readmitted_binary']
    X = df.drop(columns=['readmitted_binary'])
    
    folds = list(sgkf.split(X, y, groups))
    
    # fold 0 is test
    test_idx = folds[0][1]
    
    # use fold 1 for validation
    val_idx = folds[1][1]
    
    # the rest is train
    train_idx = []
    for i in range(2, 10):
        train_idx.extend(folds[i][1])
        
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    
    return df_train, df_val, df_test

def build_preprocessors(df_train: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Ensure dense output for OneHotEncoder, use sparse_output=False for newer sklearn, sparse=False for older
    # We'll rely on the environment having newer sklearn (1.2+) since we didn't pin versions, 
    # but we can try/except or just use sparse_output=False
    try:
        cat_pipeline_onehot = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    except TypeError:
        cat_pipeline_onehot = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
    
    preprocessor_onehot = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline_onehot, categorical_cols)
    ], remainder='drop')
    
    # OrdinalEncoder for FT-Transformer
    # We want unknown index to be valid (cardinalities + 1)
    # The OrdinalEncoder `handle_unknown="use_encoded_value"` with `unknown_value=-1`
    cat_pipeline_ft = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor_ft = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline_ft, categorical_cols)
    ], remainder='drop')
    
    # Fit on train data
    preprocessor_onehot.fit(df_train)
    preprocessor_ft.fit(df_train)
    
    # For FT-transformer, we want positive integers and handle -1 as max_cardinality
    # We can handle the -1 -> max_cardinality logic inside the torch dataset, 
    # but let's extract cardinalities now.
    
    return preprocessor_onehot, preprocessor_ft
