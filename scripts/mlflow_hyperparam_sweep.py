#!/usr/bin/env python3
"""
MLflow hyperparameter sweep — SGDClassifier on Rakuten TF-IDF features.

Runs 15 experiments with varying alpha, loss, max_iter, and max_features,
logging each run to MLflow for comparison.

Usage (local):
    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/mlflow_hyperparam_sweep.py

Usage (on VPS via Docker):
    docker compose run --rm trainer python scripts/mlflow_hyperparam_sweep.py
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
import scipy.sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid — 15 runs covering the most impactful hyperparameters
# ---------------------------------------------------------------------------
RUNS = [
    # alpha sweep (most impactful param)
    {"alpha": 1e-6,  "loss": "log_loss",       "max_iter": 1000, "label": "alpha=1e-6"},
    {"alpha": 5e-6,  "loss": "log_loss",       "max_iter": 1000, "label": "alpha=5e-6"},
    {"alpha": 1e-5,  "loss": "log_loss",       "max_iter": 1000, "label": "alpha=1e-5 (baseline)"},
    {"alpha": 2e-5,  "loss": "log_loss",       "max_iter": 1000, "label": "alpha=2e-5"},
    {"alpha": 5e-5,  "loss": "log_loss",       "max_iter": 1000, "label": "alpha=5e-5"},
    {"alpha": 1e-4,  "loss": "log_loss",       "max_iter": 1000, "label": "alpha=1e-4"},
    {"alpha": 5e-4,  "loss": "log_loss",       "max_iter": 1000, "label": "alpha=5e-4"},
    {"alpha": 1e-3,  "loss": "log_loss",       "max_iter": 1000, "label": "alpha=1e-3"},
    # loss function comparison
    {"alpha": 1e-5,  "loss": "modified_huber", "max_iter": 1000, "label": "modified_huber alpha=1e-6"},
    {"alpha": 1e-4,  "loss": "modified_huber", "max_iter": 1000, "label": "modified_huber alpha=1e-4"},
    {"alpha": 5e-4,  "loss": "modified_huber", "max_iter": 1000, "label": "modified_huber alpha=5e-4"},
    # max_iter sweep
    {"alpha": 1e-5,  "loss": "log_loss",       "max_iter": 300,  "label": "max_iter=300"},
    {"alpha": 1e-5,  "loss": "log_loss",       "max_iter": 500,  "label": "max_iter=500"},
    {"alpha": 1e-5,  "loss": "log_loss",       "max_iter": 2000, "label": "max_iter=2000"},
    # without class_weight (ablation)
    {"alpha": 1e-5,  "loss": "log_loss",       "max_iter": 1000, "label": "no_class_weight",
     "class_weight": None},
]

EXPERIMENT_NAME = "rakuten-text-baseline"
INPUT_DIR = Path(os.getenv("INPUT_DIR", "data/preprocessed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))


def _load_data():
    x_train = scipy.sparse.load_npz(INPUT_DIR / "X_train_tf.npz")
    x_val   = scipy.sparse.load_npz(INPUT_DIR / "X_val_tf.npz")

    def _load_target(path):
        df = pd.read_csv(path)
        col = "prdtypecode" if "prdtypecode" in df.columns else df.columns[0]
        return df[col]

    y_train = _load_target(INPUT_DIR / "Y_train_clean.csv")
    y_val   = _load_target(INPUT_DIR / "Y_val_clean.csv")
    return x_train, x_val, y_train, y_val


def main():
    try:
        import mlflow
    except ImportError:
        log.error("mlflow not installed — pip install mlflow")
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        log.error("Set MLFLOW_TRACKING_URI before running this script.")
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    log.info("Loading feature matrices from %s ...", INPUT_DIR)
    x_train, x_val, y_train, y_val = _load_data()
    log.info("  X_train: %s   X_val: %s", x_train.shape, x_val.shape)

    results = []
    for i, cfg in enumerate(RUNS, 1):
        run_label = cfg.pop("label")
        class_weight = cfg.pop("class_weight", "balanced")
        log.info("[%d/%d] %s ...", i, len(RUNS), run_label)

        clf = SGDClassifier(
            random_state=42,
            class_weight=class_weight,
            tol=1e-3,
            **cfg,
        )
        clf.fit(x_train, y_train)
        preds = clf.predict(x_val)

        metrics = {
            "val_accuracy": round(float(accuracy_score(y_val, preds)), 6),
            "val_macro_f1": round(float(f1_score(y_val, preds, average="macro")), 6),
            "train_samples": float(x_train.shape[0]),
            "val_samples": float(x_val.shape[0]),
        }
        params = {
            "alpha": cfg["alpha"],
            "loss": cfg["loss"],
            "max_iter": cfg["max_iter"],
            "class_weight": str(class_weight),
            "tol": 1e-3,
        }

        with mlflow.start_run(run_name=run_label):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

        log.info(
            "  -> val_accuracy=%.4f  val_macro_f1=%.4f",
            metrics["val_accuracy"], metrics["val_macro_f1"],
        )
        results.append({"run": run_label, **metrics})
        # restore for next iteration
        cfg["label"] = run_label
        cfg["class_weight"] = class_weight

    # Summary
    best = max(results, key=lambda r: r["val_accuracy"])
    log.info("=" * 60)
    log.info("SWEEP COMPLETE — %d runs logged to MLflow", len(results))
    log.info("Best run: %s (accuracy=%.4f, f1=%.4f)",
             best["run"], best["val_accuracy"], best["val_macro_f1"])
    log.info("=" * 60)

    # Save summary locally
    Path("reports").mkdir(exist_ok=True)
    summary_path = Path("reports/sweep_summary.json")
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
