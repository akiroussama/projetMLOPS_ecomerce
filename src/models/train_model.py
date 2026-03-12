from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.features.build_features import build_feature_matrices


TARGET_COLUMN = "prdtypecode"


def _load_mlflow() -> Any | None:
    try:
        import mlflow
    except ImportError:
        return None

    return mlflow


def _load_targets(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if TARGET_COLUMN in frame.columns:
        return frame[TARGET_COLUMN]
    if len(frame.columns) == 1:
        return frame.iloc[:, 0]
    raise ValueError(f"Could not resolve target column in {path}.")


@dataclass
class TrainingSummary:
    metrics: dict[str, float]
    model_path: Path
    vectorizer_path: Path
    metrics_path: Path
    report_path: Path


def train_baseline_model(
    *,
    input_dir: str | Path = "data/preprocessed",
    model_dir: str | Path = "models",
    report_dir: str | Path = "reports",
    tracking_uri: str | None = None,
    experiment_name: str = "rakuten-text-baseline",
    run_name: str | None = None,
    max_features: int = 5000,
) -> TrainingSummary:
    """Train a text-only baseline model and save API-compatible artifacts."""

    logger = logging.getLogger(__name__)
    input_path = Path(input_dir)
    model_path = Path(model_dir)
    report_path = Path(report_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    feature_files = {
        input_path / "X_train_tf.npz",
        input_path / "X_val_tf.npz",
        model_path / "tfidf_vectorizer.pkl",
    }
    if not all(path.exists() for path in feature_files):
        logger.info("feature matrices missing, build them before training")
        build_feature_matrices(
            input_dir=input_path,
            output_dir=input_path,
            model_dir=model_path,
            max_features=max_features,
        )

    x_train = scipy.sparse.load_npz(input_path / "X_train_tf.npz")
    x_val = scipy.sparse.load_npz(input_path / "X_val_tf.npz")
    y_train = _load_targets(input_path / "Y_train_clean.csv")
    y_val = _load_targets(input_path / "Y_val_clean.csv")

    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(x_train, y_train)

    validation_predictions = classifier.predict(x_val)
    metrics = {
        "val_accuracy": round(float(accuracy_score(y_val, validation_predictions)), 6),
        "val_macro_f1": round(
            float(f1_score(y_val, validation_predictions, average="macro")), 6
        ),
        "train_samples": float(x_train.shape[0]),
        "validation_samples": float(x_val.shape[0]),
        "feature_count": float(x_train.shape[1]),
    }

    serialized_model_path = model_path / "baseline_model.pkl"
    with serialized_model_path.open("wb") as model_file:
        pickle.dump(classifier, model_file)

    metrics_output_path = report_path / "training_metrics.json"
    metrics_output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    report_output_path = report_path / "classification_report.json"
    report_output_path.write_text(
        json.dumps(
            classification_report(
                y_val, validation_predictions, output_dict=True, zero_division=0
            ),
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    resolved_tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    mlflow = _load_mlflow()
    if resolved_tracking_uri and mlflow is None:
        raise RuntimeError(
            "MLflow logging was requested but mlflow is not installed in this environment."
        )

    if resolved_tracking_uri and mlflow is not None:
        mlflow.set_tracking_uri(resolved_tracking_uri)
        mlflow.set_experiment(experiment_name)

        params = {
            "model_class": classifier.__class__.__name__,
            "loss": "log_loss",
            "alpha": 1e-5,
            "max_iter": 1000,
            "class_weight": "balanced",
            "max_features": max_features,
        }

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(serialized_model_path))
            mlflow.log_artifact(str(model_path / "tfidf_vectorizer.pkl"))
            mlflow.log_artifact(str(metrics_output_path))
            mlflow.log_artifact(str(report_output_path))

    return TrainingSummary(
        metrics=metrics,
        model_path=serialized_model_path,
        vectorizer_path=model_path / "tfidf_vectorizer.pkl",
        metrics_path=metrics_output_path,
        report_path=report_output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the baseline text classifier and log the run to MLflow."
    )
    parser.add_argument(
        "--input-dir",
        default="data/preprocessed",
        help="Directory containing the clean CSV files and sparse feature matrices.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory where the serialized baseline model will be written.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory where the evaluation reports will be written.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking server URI. When omitted, MLFLOW_TRACKING_URI is used.",
    )
    parser.add_argument(
        "--experiment-name",
        default="rakuten-text-baseline",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional MLflow run name.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features to keep if features need rebuilding.",
    )
    return parser


def main(argv: list[str] | None = None) -> TrainingSummary:
    args = build_parser().parse_args(argv)
    summary = train_baseline_model(
        input_dir=args.input_dir,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        max_features=args.max_features,
    )
    logging.getLogger(__name__).info("training complete: %s", summary.metrics)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
