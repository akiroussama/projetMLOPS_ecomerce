from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = os.getenv("AIRFLOW_PROJECT_ROOT", "/opt/airflow/project")
COMMON_ENV = {
    "PYTHONPATH": PROJECT_ROOT,
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
}
ARTIFACT_CHECK = """
python - <<'PY'
from pathlib import Path

required = [
    Path("models/baseline_model.pkl"),
    Path("models/tfidf_vectorizer.pkl"),
    Path("reports/training_metrics.json"),
]
missing = [str(path) for path in required if not path.exists()]

if missing:
    raise SystemExit(f"Missing artifacts: {missing}")

print("Artifacts ready for the API service.")
PY
""".strip()


with DAG(
    dag_id="rakuten_weekly_retraining",
    description="Download data, rebuild features, retrain the baseline model and log to MLflow.",
    start_date=datetime(2026, 3, 10),
    schedule="0 2 * * 1",
    catchup=False,
    default_args={
        "owner": "mlops-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "retraining"],
) as dag:
    download_raw_data = BashOperator(
        task_id="download_raw_data",
        bash_command="python -m src.data.import_raw_data --output-dir data/raw",
        cwd=PROJECT_ROOT,
        env=COMMON_ENV,
    )

    prepare_dataset = BashOperator(
        task_id="prepare_dataset",
        bash_command="python -m src.data.make_dataset data/raw data/preprocessed",
        cwd=PROJECT_ROOT,
        env=COMMON_ENV,
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=(
            "python -m src.features.build_features "
            "--input-dir data/preprocessed "
            "--output-dir data/preprocessed "
            "--model-dir models"
        ),
        cwd=PROJECT_ROOT,
        env=COMMON_ENV,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            "python -m src.models.train_model "
            "--input-dir data/preprocessed "
            "--model-dir models "
            "--report-dir reports "
            "--experiment-name rakuten-text-baseline "
            "--run-name airflow-retraining"
        ),
        cwd=PROJECT_ROOT,
        env=COMMON_ENV,
    )

    verify_artifacts = BashOperator(
        task_id="verify_artifacts",
        bash_command=ARTIFACT_CHECK,
        cwd=PROJECT_ROOT,
        env=COMMON_ENV,
    )

    generate_drift_report = BashOperator(
        task_id="generate_drift_report",
        bash_command=(
            "python -m src.monitoring.drift_report "
            "--reference-path data/preprocessed/X_train_clean.csv "
            "--current-path data/preprocessed/X_val_clean.csv "
            "--output-dir reports/drift"
        ),
        cwd=PROJECT_ROOT,
        env=COMMON_ENV,
    )

    download_raw_data >> prepare_dataset >> build_features >> train_model >> verify_artifacts >> generate_drift_report
