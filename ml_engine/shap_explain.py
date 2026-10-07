"""
SHAP Explainability & Risk Factor Attribution Module for AvertCare
Computes global feature importances and patient-level risk factor explanations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


class AvertCareExplainer:
    """
    SHAP-based Explainer for Readmission Risk Models.
    """

    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """
        Initializes the appropriate SHAP explainer based on model type.
        """
        model_type = type(self.model).__name__
        if "RandomForest" in model_type or "Tree" in model_type or "GradientBoosting" in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Fallback to KernelExplainer or LinearExplainer
            self.explainer = shap.Explainer(self.model.predict_proba, feature_names=self.feature_names)

    def compute_shap_values(self, X):
        """
        Computes SHAP values for dataset X.
        """
        if isinstance(self.explainer, shap.TreeExplainer):
            shap_vals = self.explainer.shap_values(X)
            # If multi-class or list output, pick positive class (index 1)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            elif len(shap_vals.shape) == 3:
                shap_vals = shap_vals[:, :, 1]
        else:
            shap_vals = self.explainer(X)
            if hasattr(shap_vals, "values"):
                shap_vals = shap_vals.values
                if len(shap_vals.shape) == 3:
                    shap_vals = shap_vals[:, :, 1]

        return shap_vals

    def generate_summary_plots(self, X, output_dir="ml_engine/shap_output", max_features=20):
        """
        Generates and saves global SHAP summary bar & dot plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        shap_vals = self.compute_shap_values(X)

        feature_names = self.feature_names
        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)

        # 1. SHAP Summary Bar Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_vals,
            X,
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_features,
            show=False,
        )
        plt.title("SHAP Global Feature Importance — Readmission Risk", fontsize=12, fontweight="bold")
        plt.tight_layout()
        bar_path = os.path.join(output_dir, "shap_summary_bar.png")
        plt.savefig(bar_path, dpi=300)
        plt.close(fig)

        # 2. SHAP Summary Dot/Beeswarm Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_vals,
            X,
            feature_names=feature_names,
            plot_type="dot",
            max_display=max_features,
            show=False,
        )
        plt.title("SHAP Feature Impact Beeswarm — Readmission Risk", fontsize=12, fontweight="bold")
        plt.tight_layout()
        dot_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
        plt.savefig(dot_path, dpi=300)
        plt.close(fig)

        return {"bar_plot": bar_path, "beeswarm_plot": dot_path}

    def explain_patient_instance(self, x_single, top_k=5):
        """
        Explains risk factors for a single patient record.

        Parameters
        ----------
        x_single : np.ndarray or pd.DataFrame, shape (1, D)
            Preprocessed single patient feature vector.
        top_k : int, default=5
            Number of top risk drivers to return.

        Returns
        -------
        dict
            Dictionary containing positive risk drivers and protective factors.
        """
        if isinstance(x_single, pd.DataFrame):
            x_arr = x_single.values
        else:
            x_arr = np.asarray(x_single)

        if len(x_arr.shape) == 1:
            x_arr = x_arr.reshape(1, -1)

        shap_vals = self.compute_shap_values(x_arr)[0]
        feature_names = self.feature_names if self.feature_names is not None else [f"feature_{i}" for i in range(len(shap_vals))]

        feature_impacts = []
        for name, val, f_val in zip(feature_names, shap_vals, x_arr[0]):
            feature_impacts.append({
                "feature": name,
                "shap_value": float(val),
                "feature_value": float(f_val),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            })

        # Sort by absolute impact
        sorted_impacts = sorted(feature_impacts, key=lambda item: abs(item["shap_value"]), reverse=True)

        risk_drivers = [item for item in sorted_impacts if item["shap_value"] > 0][:top_k]
        protective_factors = [item for item in sorted_impacts if item["shap_value"] < 0][:top_k]

        return {
            "top_risk_drivers": risk_drivers,
            "top_protective_factors": protective_factors,
            "all_ranked_features": sorted_impacts[:top_k * 2],
        }
