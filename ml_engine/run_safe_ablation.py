"""Leakage-safe ablation and model selection.

Selects the neighborhood size and the model on the validation split.
Test metrics are reported after that choice. The previous leaked
train_with_rag.csv is not read.

Run from the repository root:
    python ml_engine/run_safe_ablation.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import compute_metrics, find_optimal_threshold  # noqa: E402

DATA_DIR = "ml_engine/data/processed"
RAG_CSV = "data/processed/rag_features_safe.csv"
BACKEND_DIR = "backend/models"
REPORT_PATH = "ml_engine/reports/safe_rag_ablation.md"
METRICS_PATH = "ml_engine/metrics/safe_ablation_metrics.json"
SHAP_DIR = "ml_engine/shap_output"


def _as_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _load():
    print("Loading splits and leakage-safe RAG features...", flush=True)
    frames = {}
    for name in ("train", "val", "test"):
        frames[name] = pd.read_csv(os.path.join(DATA_DIR, f"{name}_with_ids.csv"))
    rag = pd.read_csv(RAG_CSV)
    for name, df in frames.items():
        merged = df.merge(
            rag.drop(columns=["patient_nbr", "split"], errors="ignore"),
            on="encounter_id",
            how="left",
        )
        if merged["rag_readmit_rate_k10"].isna().any():
            missing = int(merged["rag_readmit_rate_k10"].isna().sum())
            raise RuntimeError(f"{name} has {missing} encounters without a safe RAG feature.")
        corr = merged["rag_readmit_rate_k10"].corr(merged["readmitted_binary"])
        print(f"  {name:5s} rows={len(merged):6d}  corr(k10, label)={corr:.4f}", flush=True)
        if abs(corr) > 0.7:
            raise RuntimeError(f"Refusing to train: {name} RAG correlation {corr:.3f} looks leaked.")
        frames[name] = merged

    prep_obj = joblib.load(os.path.join(DATA_DIR, "preprocessor_onehot.joblib"))
    preprocessor = prep_obj["preprocessor"] if isinstance(prep_obj, dict) else prep_obj
    return frames["train"], frames["val"], frames["test"], preprocessor


def _scores(model, X_val, y_val, X_test, y_test):
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    threshold = find_optimal_threshold(y_val, p_val, metric="f1")
    return {
        "val": compute_metrics(y_val, p_val, threshold=threshold),
        "val_auroc": float(compute_metrics(y_val, p_val)["auroc"]),
        "test": compute_metrics(y_test, p_test, threshold=threshold),
        "threshold": threshold,
        "p_val": p_val,
        "p_test": p_test,
    }


def _fit_xgb(X_train, y_train, max_depth, use_scale):
    """Early-stop on an internal slice of train, then refit on all of train.

    The official validation split is not used here, so validation AUROC stays
    a fair comparison against the other model families.
    """
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    rng = np.random.default_rng(42)
    es_idx = np.concatenate([
        rng.choice(pos_idx, size=max(1, int(0.1 * len(pos_idx))), replace=False),
        rng.choice(neg_idx, size=max(1, int(0.1 * len(neg_idx))), replace=False),
    ])
    fit_mask = np.ones(len(y_train), dtype=bool)
    fit_mask[es_idx] = False
    pos = max(float(y_train[fit_mask].sum()), 1.0)
    spw = ((int(fit_mask.sum()) - pos) / pos) if use_scale else 1.0
    probe = XGBClassifier(
        n_estimators=500,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )
    probe.fit(X_train[fit_mask], y_train[fit_mask], eval_set=[(X_train[es_idx], y_train[es_idx])], verbose=False)
    n_trees = int(probe.best_iteration) + 1
    model = XGBClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def _candidates(X_train, y_train, X_val, y_val):
    pos = max(float(y_train.sum()), 1.0)
    spw = (len(y_train) - pos) / pos
    built = []

    for C in (0.2, 1.0, 5.0):
        built.append((
            f"LogisticRegression_C{C}",
            LogisticRegression(
                C=C, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42
            ),
        ))

    built.append((
        "RandomForest_d16",
        RandomForestClassifier(
            n_estimators=250, max_depth=16, min_samples_leaf=4,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        ),
    ))
    built.append((
        "RandomForest_d24",
        RandomForestClassifier(
            n_estimators=350, max_depth=24, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        ),
    ))

    for depth, leaves in ((6, 31), (None, 63)):
        label = f"HistGradientBoosting_d{depth}_l{leaves}"
        built.append((
            label,
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_depth=depth,
                max_leaf_nodes=leaves,
                max_iter=400,
                l2_regularization=0.1,
                early_stopping=True,
                n_iter_no_change=25,
                validation_fraction=0.1,
                class_weight="balanced",
                random_state=42,
            ),
        ))

    for depth, scaled in ((4, True), (6, True), (6, False), (8, True)):
        built.append((
            f"XGBoost_d{depth}_{'spw' if scaled else 'unweighted'}",
            ("xgb", depth, scaled),
        ))

    # spw is recorded so the report can show the imbalance weight used by XGB.
    _ = spw
    fitted = []
    for name, spec in built:
        print(f"    fitting {name}...", flush=True)
        if isinstance(spec, tuple) and spec[0] == "xgb":
            model = _fit_xgb(X_train, y_train, spec[1], spec[2])
        else:
            model = spec
            model.fit(X_train, y_train)
        fitted.append((name, model))
    return fitted


def _family(name: str) -> str:
    for family in ("LogisticRegression", "RandomForest", "HistGradientBoosting", "XGBoost"):
        if name.startswith(family):
            return family
    return name


def _pick_k(train, val, test, X_train, X_val, X_test, y_train, y_val):
    print("\nChoosing neighborhood size on validation AUROC...", flush=True)
    options = {"tabular": None}
    for k in (5, 10, 20):
        options[f"k{k}"] = k
    scores = {}
    for label, k in options.items():
        def attach(X, df, k=k):
            if k is None:
                return X
            col = df[f"rag_readmit_rate_k{k}"].to_numpy(dtype=np.float32).reshape(-1, 1)
            return np.hstack([X, col])

        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_depth=8, max_leaf_nodes=31, max_iter=250,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
            class_weight="balanced", random_state=42,
        )
        model.fit(attach(X_train, train), y_train)
        p_val = model.predict_proba(attach(X_val, val))[:, 1]
        auroc = float(compute_metrics(y_val, p_val)["auroc"])
        scores[label] = auroc
        print(f"  probe {label:8s} val AUROC {auroc:.4f}", flush=True)
    best_label = max(scores, key=scores.get)
    return options[best_label], scores


def _write_report(rows, winner, k, probe_scores):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = [
        "# Leakage-safe RAG ablation",
        "",
        "Neighborhood rates were computed from training encounters only.",
        "Training queries excluded every encounter of the same patient.",
        "Validation and test patients were never in the index.",
        "Notes contain discharge-time fields only.",
        "",
        "## Neighborhood-size probe (validation AUROC, one boosted tree)",
        "",
    ]
    for label, score in probe_scores.items():
        lines.append(f"- {label}: {score:.4f}")
    chosen = "tabular only" if k is None else f"k={k}"
    lines += [
        "",
        f"Feature set used for the full grid: **{chosen}**, plus a tabular-only grid for the comparison.",
        "",
        "Winner rule: highest validation AUROC, then validation AUPRC. Test metrics are confirmatory.",
        "",
        "| Model | Features | Val AUROC | Val AUPRC | Test AUROC | Test AUPRC | Test F1 | Test Precision | Test Recall | Threshold |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['features']} | {row['val_auroc']:.4f} | {row['val_auprc']:.4f} | "
            f"{row['test_auroc']:.4f} | {row['test_auprc']:.4f} | {row['test_f1']:.4f} | "
            f"{row['test_precision']:.4f} | {row['test_recall']:.4f} | {row['threshold']:.2f} |"
        )
    lines += [
        "",
        f"## Winner",
        "",
        f"- Model: `{winner['model']}`",
        f"- Features: {winner['features']}",
        f"- Validation AUROC: {winner['val_auroc']:.4f}",
        f"- Test AUROC: {winner['test_auroc']:.4f}",
        f"- Test AUPRC: {winner['test_auprc']:.4f}",
        "",
        "The previous 0.6971 result used notes that contained the readmission label and an index that mixed held-out encounters. It is not comparable to this table.",
        "",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote {REPORT_PATH}", flush=True)


def _export(winner_model, preprocessor, feature_names, uses_rag, k, df_test, y_test, X_test, threshold):
    os.makedirs(BACKEND_DIR, exist_ok=True)
    current = os.path.join(BACKEND_DIR, "best_model.joblib")
    backup = os.path.join(BACKEND_DIR, "best_model_leaked_backup.joblib")
    if os.path.exists(current) and not os.path.exists(backup):
        shutil.copy2(current, backup)
        print(f"Backed up previous model -> {backup}", flush=True)

    model_path = os.path.join(BACKEND_DIR, "best_model.joblib")
    joblib.dump(winner_model, model_path)
    shutil.copy2(os.path.join(DATA_DIR, "preprocessor_onehot.joblib"), os.path.join(BACKEND_DIR, "preprocessor.joblib"))

    meta = {
        "feature_names": feature_names,
        "uses_rag": uses_rag,
        "rag_feature_column": "rag_readmit_rate" if uses_rag else None,
        "rag_k": int(k) if k else None,
        "decision_threshold": threshold,
        "winning_model": winner_model.__class__.__name__,
        "leakage_safe": True,
    }
    with open(os.path.join(BACKEND_DIR, "feature_names.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    rng = np.random.default_rng(42)
    idx = int(rng.integers(0, len(df_test)))
    prob = float(winner_model.predict_proba(X_test[idx:idx + 1])[:, 1][0])
    sample = {
        "encounter_id": int(df_test.iloc[idx]["encounter_id"]),
        "ground_truth_readmitted_binary": int(y_test[idx]),
        "model_input_rag_readmit_rate": None if k is None else float(df_test.iloc[idx][f"rag_readmit_rate_k{k}"]),
        "expected_probability": round(prob, 4),
        "expected_prediction": int(prob >= threshold),
        "decision_threshold_used": threshold,
        "winning_model": winner_model.__class__.__name__,
        "leakage_safe": True,
    }
    with open(os.path.join(BACKEND_DIR, "sample_test_patient.json"), "w", encoding="utf-8") as handle:
        json.dump(sample, handle, indent=2)
    print(f"Exported {model_path}", flush=True)


def _shap(model, X_test, feature_names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import shap
        os.makedirs(SHAP_DIR, exist_ok=True)
        rng = np.random.default_rng(42)
        take = min(400, X_test.shape[0])
        sample = X_test[rng.choice(X_test.shape[0], size=take, replace=False)]
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            values = values[1]
        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, 1]
        shap.summary_plot(values, sample, feature_names=feature_names, show=False, max_display=20, plot_type="bar")
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, "shap_summary_bar.png"), dpi=140)
        plt.close()
        print(f"Wrote {SHAP_DIR}/shap_summary_bar.png", flush=True)
    except Exception as exc:
        print(f"SHAP plot skipped: {exc}", flush=True)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    train, val, test, preprocessor = _load()
    y_train = train["readmitted_binary"].to_numpy()
    y_val = val["readmitted_binary"].to_numpy()
    y_test = test["readmitted_binary"].to_numpy()

    print("Transforming tabular features...", flush=True)
    X_train = _as_dense(preprocessor.transform(train))
    X_val = _as_dense(preprocessor.transform(val))
    X_test = _as_dense(preprocessor.transform(test))
    print(f"  tabular dim {X_train.shape[1]}", flush=True)

    k, probe_scores = _pick_k(train, val, test, X_train, X_val, X_test, y_train, y_val)

    feature_sets = [("tabular", None)]
    if k is not None:
        feature_sets.append((f"tabular+rag_k{k}", k))
    else:
        # Still score k=10 so the report shows the RAG comparison.
        feature_sets.append(("tabular+rag_k10", 10))
        k = 10

    rows = []
    best_tab = None
    best_rag = None
    for feature_label, feature_k in feature_sets:
        print(f"\nFull grid: {feature_label}", flush=True)

        def attach(X, df, feature_k=feature_k):
            if feature_k is None:
                return X
            col = df[f"rag_readmit_rate_k{feature_k}"].to_numpy(dtype=np.float32).reshape(-1, 1)
            return np.hstack([X, col])

        Xtr, Xva, Xte = attach(X_train, train), attach(X_val, val), attach(X_test, test)
        for name, model in _candidates(Xtr, y_train, Xva, y_val):
            scored = _scores(model, Xva, y_val, Xte, y_test)
            row = {
                "model": name,
                "family": _family(name),
                "features": feature_label,
                "val_auroc": scored["val_auroc"],
                "val_auprc": scored["val"]["auprc"],
                "test_auroc": scored["test"]["auroc"],
                "test_auprc": scored["test"]["auprc"],
                "test_f1": scored["test"]["f1_score"],
                "test_precision": scored["test"]["precision"],
                "test_recall": scored["test"]["recall_sensitivity"],
                "threshold": scored["threshold"],
            }
            rows.append(row)
            print(
                f"    {name} val AUROC {row['val_auroc']:.4f}  test AUROC {row['test_auroc']:.4f}",
                flush=True,
            )
            pack = (row, model, Xte, feature_k)
            rank = (row["val_auroc"], row["val_auprc"])
            if feature_k is None:
                if best_tab is None or rank > (best_tab[0]["val_auroc"], best_tab[0]["val_auprc"]):
                    best_tab = pack
            elif best_rag is None or rank > (best_rag[0]["val_auroc"], best_rag[0]["val_auprc"]):
                best_rag = pack

    packs = [pack for pack in (best_tab, best_rag) if pack is not None]
    winner_row, winner_model, winner_X_test, winner_k = max(
        packs, key=lambda pack: (pack[0]["val_auroc"], pack[0]["val_auprc"])
    )
    uses_rag = winner_row["features"] != "tabular"

    tab_names = list(preprocessor.get_feature_names_out())
    feature_names = tab_names + (["rag_readmit_rate"] if uses_rag else [])
    _export(winner_model, preprocessor, feature_names, uses_rag, winner_k if uses_rag else None, test, y_test, winner_X_test, winner_row["threshold"])
    _shap(winner_model, winner_X_test, feature_names)
    _write_report(rows, winner_row, k, probe_scores)

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump({"probe": probe_scores, "rows": rows, "winner": winner_row}, handle, indent=2)
    print(
        f"\nWinner {winner_row['model']} ({winner_row['features']}) "
        f"val AUROC {winner_row['val_auroc']:.4f} test AUROC {winner_row['test_auroc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
