"""
RAG Integration Layer for AvertCare Readmission Risk System
Interface for integrating M2 RAG vector embeddings & clinical text features
with M1 Tabular ML Pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib


class RAGFeatureIntegrator:
    """
    RAG Integration Layer.

    Designed for clean integration when M2 delivers clinical RAG outputs:
    1. Vector Embeddings (e.g., BioBERT, Med-PaLM, or sentence-transformer embeddings from discharge notes).
    2. Structured RAG risk signals (e.g., clinical sentiment, LLM-extracted risk severity scores).

    Maintains zero dependency on fake data during M1 standalone training.
    """

    def __init__(self, tabular_preprocessor_path=None, model_path=None):
        self.tabular_preprocessor_path = tabular_preprocessor_path
        self.model_path = model_path
        self.tabular_preprocessor = None
        self.model = None

        if tabular_preprocessor_path and os.path.exists(tabular_preprocessor_path):
            self.tabular_preprocessor = joblib.load(tabular_preprocessor_path)

        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    @staticmethod
    def concatenate_features(X_tabular: np.ndarray, X_rag: np.ndarray) -> np.ndarray:
        """
        Combines M1 preprocessed tabular features with M2 RAG feature vectors.

        Parameters
        ----------
        X_tabular : np.ndarray, shape (N, D_tab)
            Preprocessed tabular feature matrix from M1 preprocessor.
        X_rag : np.ndarray, shape (N, D_rag)
            Clinical text embeddings / RAG features matrix from M2.

        Returns
        -------
        np.ndarray, shape (N, D_tab + D_rag)
            Combined feature representation.
        """
        if X_tabular.shape[0] != X_rag.shape[0]:
            raise ValueError(
                f"Row mismatch: X_tabular has {X_tabular.shape[0]} rows, "
                f"but X_rag has {X_rag.shape[0]} rows."
            )
        return np.hstack([X_tabular, X_rag])

    def fit_combined_model(self, model_class, X_tabular: np.ndarray, X_rag: np.ndarray, y: np.ndarray, **kwargs):
        """
        Trains a joint ML model on combined tabular + RAG features.

        Parameters
        ----------
        model_class : estimator class (e.g., RandomForestClassifier)
            Scikit-learn compatible classifier.
        X_tabular : np.ndarray
        X_rag : np.ndarray
        y : np.ndarray
        **kwargs : hyperparameter arguments for model_class instantiation.

        Returns
        -------
        trained model instance
        """
        X_combined = self.concatenate_features(X_tabular, X_rag)
        model = model_class(**kwargs)
        model.fit(X_combined, y)
        self.model = model
        return model

    def predict_combined_risk(self, df_patient_raw: pd.DataFrame, X_rag: np.ndarray) -> np.ndarray:
        """
        Inference interface for combined Tabular + RAG patient records.

        Parameters
        ----------
        df_patient_raw : pd.DataFrame
            Raw patient record containing M1 features.
        X_rag : np.ndarray
            RAG embeddings vector for the patient from M2.

        Returns
        -------
        np.ndarray
            Predicted readmission risk probabilities.
        """
        if self.tabular_preprocessor is None:
            raise ValueError("Tabular preprocessor is not loaded.")
        if self.model is None:
            raise ValueError("Joint RAG model is not loaded.")

        X_tab = self.tabular_preprocessor.transform(df_patient_raw)
        X_combined = self.concatenate_features(X_tab, X_rag)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_combined)[:, 1]
        else:
            probs = self.model.predict(X_combined)

        return probs

    def get_integration_spec(self) -> dict:
        """
        Returns formal interface specification for M2 RAG team handoff.
        """
        return {
            "status": "Ready for M2 RAG integration",
            "tabular_feature_dim": 144,
            "expected_rag_input": "Matrix of shape (N, D_rag) where D_rag is text embedding dimension",
            "supported_fusion_strategies": ["Early fusion (feature concatenation)", "Late fusion (probability blending)"],
            "handoff_checkpoint_path": self.model_path,
        }
