#!/usr/bin/env bash
# ------------------------------------------------------------------
# bootstrap.sh  -  Train the baseline model from scratch (S3 -> model)
#
# This script replaces the need for DVC / Google Drive.
# It downloads raw data from the public S3 bucket, preprocesses it,
# builds TF-IDF features, and trains the SGDClassifier baseline.
#
# Usage:
#   Local:   bash scripts/bootstrap.sh
#   Docker:  docker compose run trainer bash scripts/bootstrap.sh
# ------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="${AIRFLOW_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

echo "========================================="
echo " MLOps Bootstrap - Train from scratch"
echo "========================================="

echo ""
echo "[1/4] Downloading raw data from S3..."
python -m src.data.import_raw_data --output-dir data/raw

echo ""
echo "[2/4] Cleaning and splitting dataset..."
python -m src.data.make_dataset data/raw data/preprocessed

echo ""
echo "[3/4] Building TF-IDF features..."
python -m src.features.build_features \
    --input-dir data/preprocessed \
    --output-dir data/preprocessed \
    --model-dir models

echo ""
echo "[4/4] Training SGDClassifier baseline..."
python -m src.models.train_model \
    --input-dir data/preprocessed \
    --model-dir models \
    --report-dir reports \
    --experiment-name rakuten-text-baseline \
    --run-name bootstrap-$(date +%Y%m%d-%H%M%S)

echo ""
echo "========================================="
echo " Bootstrap complete!"
echo " Artifacts in: models/"
echo "========================================="
ls -lh models/*.pkl 2>/dev/null || echo "(no .pkl files found)"
