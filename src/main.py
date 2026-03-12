from __future__ import annotations

import argparse
import logging

from src.data.import_raw_data import import_raw_data
from src.data.make_dataset import prepare_datasets
from src.features.build_features import build_feature_matrices
from src.models.train_model import train_baseline_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end baseline training pipeline."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory where raw CSV files are stored or downloaded.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        default="data/preprocessed",
        help="Directory where cleaned datasets and sparse features are written.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory where serialized model artifacts are written.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory where training reports are written.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse the local raw CSV files instead of downloading them first.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. When omitted, MLFLOW_TRACKING_URI is used.",
    )
    parser.add_argument(
        "--experiment-name",
        default="rakuten-text-baseline",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features to keep.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    logger = logging.getLogger(__name__)

    if not args.skip_download:
        import_raw_data(raw_data_relative_path=args.raw_dir)

    dataset_summary = prepare_datasets(args.raw_dir, args.preprocessed_dir)
    feature_summary = build_feature_matrices(
        input_dir=args.preprocessed_dir,
        output_dir=args.preprocessed_dir,
        model_dir=args.model_dir,
        max_features=args.max_features,
    )
    training_summary = train_baseline_model(
        input_dir=args.preprocessed_dir,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        max_features=args.max_features,
    )

    summary = {
        "dataset": dataset_summary,
        "features": feature_summary,
        "metrics": training_summary.metrics,
    }
    logger.info("pipeline complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
