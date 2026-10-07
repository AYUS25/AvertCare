"""
Inference Pipeline & Handoff Interface for AvertCare
Transforms raw patient records, computes readmission risk, categorizes risk bands,
and extracts top clinical risk drivers for backend API handoff.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


RISK_BANDS = {
    "LOW": {"min": 0.0, "max": 0.20, "label": "Low Risk", "color": "#10B981"},
    "MODERATE": {"min": 0.20, "max": 0.50, "label": "Moderate Risk", "color": "#F59E0B"},
    "HIGH": {"min": 0.50, "max": 1.0, "label": "High Risk", "color": "#EF4444"},
}


class AvertCarePredictor:
    """
    Production Predictor for AvertCare Hospital Readmission System.
    """

    def __init__(
        self,
        preprocessor_path="ml_engine/data/processed/preprocessor_onehot.joblib",
        model_path="ml_engine/models/random_forest.joblib",
    ):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path
        self.preprocessor_dict = None
        self.preprocessor = None
        self.model = None
        self.feature_names = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads preprocessor and model artifacts."""
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {self.preprocessor_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

        self.preprocessor_dict = joblib.load(self.preprocessor_path)
        if isinstance(self.preprocessor_dict, dict) and "preprocessor" in self.preprocessor_dict:
            self.preprocessor = self.preprocessor_dict["preprocessor"]
            self.feature_names = self.preprocessor_dict.get("feature_names")
        else:
            self.preprocessor = self.preprocessor_dict

        self.model = joblib.load(self.model_path)

        if self.feature_names is None and hasattr(self.preprocessor, "get_feature_names_out"):
            try:
                self.feature_names = list(self.preprocessor.get_feature_names_out())
            except Exception:
                pass

    @staticmethod
    def _categorize_risk_band(probability: float) -> dict:
        """Categorizes probability into Low, Moderate, or High risk band."""
        if probability < 0.20:
            band_key = "LOW"
        elif probability <= 0.50:
            band_key = "MODERATE"
        else:
            band_key = "HIGH"

        info = RISK_BANDS[band_key]
        return {
            "risk_band": band_key,
            "risk_label": info["label"],
            "color_code": info["color"],
        }

    def predict_patient_dict(self, patient_dict: dict, top_k_drivers: int = 5) -> dict:
        """
        Predicts readmission risk for a single patient raw dictionary.

        Parameters
        ----------
        patient_dict : dict
            Dictionary of raw patient feature fields.
        top_k_drivers : int, default=5
            Number of top risk factors to extract.

        Returns
        -------
        dict
            Structured JSON-friendly inference dictionary.
        """
        df_patient = pd.DataFrame([patient_dict])
        return self.predict_patient_dataframe(df_patient, top_k_drivers=top_k_drivers)[0]

    def predict_patient_dataframe(self, df_raw: pd.DataFrame, top_k_drivers: int = 5) -> list:
        """
        Predicts readmission risk for a batch DataFrame of raw patient records.

        Parameters
        ----------
        df_raw : pd.DataFrame
            DataFrame of raw patient records.
        top_k_drivers : int, default=5

        Returns
        -------
        """
        from clean import clean_inference_patient

        df_clean = clean_inference_patient(df_raw)
        X_trans = self.preprocessor.transform(df_clean)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_trans)[:, 1]
        else:
            probs = self.model.predict(X_trans)

        results = []

        # Calculate feature attributions using model coefficients or feature importances
        feature_importance_weights = None
        if hasattr(self.model, "feature_importances_"):
            feature_importance_weights = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            feature_importance_weights = np.abs(self.model.coef_.ravel())

        for idx, prob in enumerate(probs):
            prob_val = float(prob)
            band_info = self._categorize_risk_band(prob_val)

            # Top feature drivers for this record
            top_drivers = []
            if feature_importance_weights is not None and len(feature_importance_weights) == X_trans.shape[1]:
                row_vals = X_trans[idx] if not hasattr(X_trans, "toarray") else X_trans[idx].toarray().ravel()
                # Compute contribution = feature_value * importance_weight
                contribs = row_vals * feature_importance_weights
                top_indices = np.argsort(np.abs(contribs))[::-1][:top_k_drivers]

                f_names = self.feature_names or [f"feature_{i}" for i in range(len(contribs))]
                for fi in top_indices:
                    top_drivers.append({
                        "feature": str(f_names[fi]),
                        "value": float(row_vals[fi]),
                        "importance": float(feature_importance_weights[fi]),
                        "contribution": float(contribs[fi]),
                    })

            results.append({
                "readmission_risk_probability": round(prob_val, 4),
                "readmission_predicted": int(prob_val >= 0.5),
                "risk_band": band_info["risk_band"],
                "risk_label": band_info["risk_label"],
                "color_code": band_info["color_code"],
                "top_risk_drivers": top_drivers,
            })

        return results


def predict_patient_risk(patient_dict: dict, model_type="random_forest") -> dict:
    """
    Convenience functional API for backend integration.
    """
    if model_type == "logistic_regression":
        model_path = "ml_engine/models/logistic_regression.joblib"
    else:
        model_path = "ml_engine/models/random_forest.joblib"

    predictor = AvertCarePredictor(
        preprocessor_path="ml_engine/data/processed/preprocessor_onehot.joblib",
        model_path=model_path,
    )
    return predictor.predict_patient_dict(patient_dict)
