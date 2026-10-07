"""
Evaluation Metrics & Visualization Module for AvertCare Readmission Risk Pipeline
Phase 3 Model Evaluation Standardized Framework
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes comprehensive evaluation metrics for binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for class 1.
    threshold : float, default=0.5
        Decision threshold for binary predictions.

    Returns
    -------
    dict
        Dictionary containing calculated metrics.
    """
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), 1e-15, 1 - 1e-15)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))  # Sensitivity
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # Specificity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    ll = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "threshold": float(threshold),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def find_optimal_threshold(y_true, y_prob, metric="f1"):
    """
    Finds optimal decision threshold on validation set.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_prob : array-like
        Predicted probabilities.
    metric : str, default='f1'
        Metric to maximize ('f1' or 'youden' for Sensitivity + Specificity - 1).

    Returns
    -------
    float
        Best threshold value.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_score = -1.0

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sens + spec - 1.0
        else:
            raise ValueError(f"Unknown metric {metric}")

        if score > best_score:
            best_score = score
            best_thresh = th

    return float(best_thresh)


def plot_roc_curves(prob_dict, y_true, save_path):
    """
    Plots overlay ROC curves for multiple models.

    Parameters
    ----------
    prob_dict : dict
        Dictionary of model_name -> y_prob array.
    y_true : array-like
        Ground truth labels.
    save_path : str
        File path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, y_prob in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{model_name} (AUROC = {auc:.3f})", lw=2)

    ax.plot([0, 1], [0, 1], "k--", label="Chance (AUROC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves — Hospital Readmission Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pr_curves(prob_dict, y_true, save_path):
    """
    Plots overlay Precision-Recall curves for multiple models.

    Parameters
    ----------
    prob_dict : dict
        Dictionary of model_name -> y_prob array.
    y_true : array-like
        Ground truth labels.
    save_path : str
        File path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    no_skill = np.mean(y_true)

    for model_name, y_prob in prob_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{model_name} (AUPRC = {auprc:.3f})", lw=2)

    ax.plot([0, 1], [no_skill, no_skill], "k--", label=f"Baseline ({no_skill:.3f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — Readmission Risk", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def save_metrics_summary(metrics_dict, save_path):
    """
    Saves metrics dictionary to a formatted JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
