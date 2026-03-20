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

_EVIDENTLY_AVAILABLE = False
try:
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.report import Report
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    pass  # Falls back to scipy KS-test implementation


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

    # --- Build drift report (Evidently if available, scipy KS-test fallback) ---
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "drift_report.html"
    json_path = output_dir / "drift_report.json"

    if _EVIDENTLY_AVAILABLE:
        print("Running Evidently drift analysis ...")
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference_features, current_data=current_features)
        report.save_html(str(html_path))
        print(f"HTML report saved to {html_path}")
        report.save_json(str(json_path))
        print(f"JSON report saved to {json_path}")
    else:
        print("Evidently not available — using scipy KS-test fallback ...")
        from scipy import stats

        feat_details: dict = {}
        drift_metrics: dict = {}
        for col in reference_features.columns:
            ks_stat, p_val = stats.ks_2samp(reference_features[col], current_features[col])
            drift = bool(p_val < 0.05)
            feat_details[col] = {
                "ks_stat": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 6),
                "drift_detected": drift,
            }
            drift_metrics[f"ks_{col}"] = round(float(ks_stat), 4)
            drift_metrics[f"pval_{col}"] = round(float(p_val), 6)
            print(f"  {col}: KS={ks_stat:.4f}, p={p_val:.6f}, {'DRIFT' if drift else 'ok'}")

        drift_share = sum(1 for v in feat_details.values() if v["drift_detected"]) / len(feat_details)
        report_data = {
            "summary": {
                "drift_share": drift_share,
                "method": "KS-test (scipy)",
                "n_reference": len(reference_features),
                "n_current": len(current_features),
            },
            "features": feat_details,
        }
        json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        print(f"JSON report saved to {json_path}")

        rows = "\n".join(
            f'<tr class="{"drift" if v["drift_detected"] else "nodrift"}">'
            f"<td>{c}</td><td>{v['ks_stat']:.4f}</td>"
            f"<td>{v['p_value']:.6f}</td>"
            f'<td>{"YES" if v["drift_detected"] else "NO"}</td></tr>'
            for c, v in feat_details.items()
        )
        html_path.write_text(
            f"""<!DOCTYPE html><html><head><title>Drift Report</title>
<style>body{{font-family:sans-serif;margin:40px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:8px}}th{{background:#4CAF50;color:white}}
.drift{{background:#ffcccc}}.nodrift{{background:#ccffcc}}</style></head><body>
<h1>Rakuten MLOps — Data Drift Report (KS-test)</h1>
<p>Reference: {len(reference_features)} rows | Current: {len(current_features)} rows</p>
<h3>Drift Share: {drift_share:.0%}</h3>
<table><tr><th>Feature</th><th>KS Stat</th><th>p-value</th><th>Drift (p&lt;0.05)</th></tr>
{rows}</table></body></html>""",
            encoding="utf-8",
        )
        print(f"HTML report saved to {html_path}")
        print(f"Drift share: {drift_share:.0%} — {'No significant drift detected' if drift_share == 0 else 'Drift detected!'}")

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
