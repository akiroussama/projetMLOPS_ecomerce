#!/usr/bin/env python3
"""
Generate an Evidently drift report and log it to MLflow as an artifact.

Usage (local):
    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/run_drift_mlflow.py

Usage (on VPS via Docker):
    docker compose run --rm trainer python scripts/run_drift_mlflow.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", "data/preprocessed/X_train_clean.csv"))
CURRENT_PATH   = Path(os.getenv("CURRENT_PATH",   "data/preprocessed/X_val_clean.csv"))
OUTPUT_DIR     = Path(os.getenv("DRIFT_OUTPUT_DIR", "reports/drift"))
EXPERIMENT_NAME = "rakuten-data-drift"


def main():
    # ── 1. Generate the Evidently report ─────────────────────────────────────
    try:
        from src.monitoring.drift_report import generate_drift_report
    except ImportError:
        log.error("Could not import src.monitoring.drift_report — check PYTHONPATH")
        return

    if not REFERENCE_PATH.exists() or not CURRENT_PATH.exists():
        log.error(
            "Data files not found: %s / %s\n"
            "Run the data preparation step first:\n"
            "  python -m src.data.make_dataset data/raw data/preprocessed",
            REFERENCE_PATH, CURRENT_PATH,
        )
        return

    log.info("Generating Evidently drift report ...")
    generate_drift_report(
        reference_path=REFERENCE_PATH,
        current_path=CURRENT_PATH,
        output_dir=OUTPUT_DIR,
    )

    html_path = OUTPUT_DIR / "drift_report.html"
    json_path = OUTPUT_DIR / "drift_report.json"

    # ── 2. Log to MLflow ─────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        log.warning(
            "MLFLOW_TRACKING_URI not set — skipping MLflow logging.\n"
            "Report saved locally at: %s", html_path,
        )
        return

    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed — report saved locally at: %s", html_path)
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    log.info("Logging drift artifacts to MLflow experiment '%s' ...", EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evidently-drift-report"):
        mlflow.log_params({
            "reference_data": str(REFERENCE_PATH),
            "current_data": str(CURRENT_PATH),
        })
        if html_path.exists():
            mlflow.log_artifact(str(html_path), artifact_path="drift")
            log.info("  Logged: %s", html_path)
        if json_path.exists():
            mlflow.log_artifact(str(json_path), artifact_path="drift")
            log.info("  Logged: %s", json_path)

    log.info("Drift report successfully logged to MLflow.")


if __name__ == "__main__":
    main()
