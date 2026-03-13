"""Generate an Evidently data-drift report comparing reference and current datasets.

Usage (standalone)::

    python -m src.monitoring.drift_report \
        --reference-path data/preprocessed/X_train_clean.csv \
        --current-path   data/preprocessed/X_val_clean.csv \
        --output-dir     reports/drift

The script derives tabular features from the raw text columns so that
Evidently can compute meaningful drift statistics:

- ``text_length``        – character length of the *designation* column
- ``has_description``    – 1 if *description* is non-empty, else 0
- ``description_length`` – character length of the *description* column
- ``word_count``         – number of whitespace-delimited words in *designation*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.report import Report
except ImportError:
    print(
        "ERROR: The 'evidently' package is not installed.\n"
        "Install monitoring dependencies with:\n\n"
        "    pip install -r requirements-monitoring.txt\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ──────────────────────────────────────────────────────────────────────

def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with derived numeric features from text columns."""
    out = pd.DataFrame()

    designation = df["designation"].fillna("")
    description = df["description"].fillna("")

    out["text_length"] = designation.str.len()
    out["has_description"] = (description.str.len() > 0).astype(int)
    out["description_length"] = description.str.len()
    out["word_count"] = designation.str.split().str.len().fillna(0).astype(int)

    return out


# ──────────────────────────────────────────────────────────────────────
# Main report generation
# ──────────────────────────────────────────────────────────────────────

def generate_drift_report(
    reference_path: str | Path,
    current_path: str | Path,
    output_dir: str | Path,
) -> None:
    """Generate HTML + JSON drift reports and write them to *output_dir*."""

    reference_path = Path(reference_path)
    current_path = Path(current_path)
    output_dir = Path(output_dir)

    # --- Validate inputs ------------------------------------------------
    if not reference_path.exists():
        print(
            f"ERROR: Reference data file not found: {reference_path}\n"
            "Make sure you have run the data preparation step first:\n\n"
            "    python -m src.data.make_dataset data/raw data/preprocessed\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not current_path.exists():
        print(
            f"ERROR: Current data file not found: {current_path}\n"
            "Make sure you have run the data preparation step first:\n\n"
            "    python -m src.data.make_dataset data/raw data/preprocessed\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load data -------------------------------------------------------
    print(f"Loading reference data from {reference_path} ...")
    reference_df = pd.read_csv(reference_path)
    print(f"  -> {len(reference_df)} rows")

    print(f"Loading current data from {current_path} ...")
    current_df = pd.read_csv(current_path)
    print(f"  -> {len(current_df)} rows")

    # --- Derive tabular features from text columns -----------------------
    print("Deriving tabular features from text columns ...")
    reference_features = _derive_features(reference_df)
    current_features = _derive_features(current_df)

    # --- Build Evidently report ------------------------------------------
    print("Running Evidently drift analysis ...")
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )
    report.run(reference_data=reference_features, current_data=current_features)

    # --- Persist outputs -------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "drift_report.html"
    json_path = output_dir / "drift_report.json"

    report.save_html(str(html_path))
    print(f"HTML report saved to {html_path}")

    report.save_json(str(json_path))
    print(f"JSON report saved to {json_path}")

    # Quick summary from the JSON
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            report_data = json.load(fh)
        print("\n--- Drift report summary ---")
        print(json.dumps(report_data.get("summary", {}), indent=2))
    except Exception:
        pass  # non-critical – the files are already written

    print("\nDrift report generation complete.")


# ──────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an Evidently data-drift report.",
    )
    parser.add_argument(
        "--reference-path",
        default="data/preprocessed/X_train_clean.csv",
        help="Path to the reference (training) CSV file.",
    )
    parser.add_argument(
        "--current-path",
        default="data/preprocessed/X_val_clean.csv",
        help="Path to the current (validation / production) CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/drift",
        help="Directory where the HTML and JSON reports will be saved.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    generate_drift_report(
        reference_path=args.reference_path,
        current_path=args.current_path,
        output_dir=args.output_dir,
    )
