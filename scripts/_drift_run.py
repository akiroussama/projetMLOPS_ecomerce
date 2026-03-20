"""Drift report using scipy KS-test + MLflow logging (evidently 0.7 compat)."""
import json, mlflow
import pandas as pd
from pathlib import Path
from scipy import stats

ref = pd.read_csv("data/preprocessed/X_train_clean.csv").fillna("")
cur = pd.read_csv("data/preprocessed/X_val_clean.csv").fillna("")

def features(df):
    out = pd.DataFrame()
    out["text_length"] = df["designation"].str.len()
    out["has_description"] = (df["description"].str.len() > 0).astype(int)
    out["description_length"] = df["description"].str.len()
    out["word_count"] = df["designation"].str.split().str.len().fillna(0).astype(int)
    return out

ref_f, cur_f = features(ref), features(cur)

drift_metrics, feat_details = {}, {}
for col in ref_f.columns:
    ks_stat, p_val = stats.ks_2samp(ref_f[col], cur_f[col])
    drift = p_val < 0.05
    drift_metrics[f"ks_{col}"] = round(float(ks_stat), 4)
    drift_metrics[f"pval_{col}"] = round(float(p_val), 6)
    drift_metrics[f"drift_{col}"] = 1 if drift else 0
    feat_details[col] = {"ks_stat": round(float(ks_stat), 4), "p_value": round(float(p_val), 6), "drift_detected": bool(drift)}
    print(f"{col}: KS={ks_stat:.4f}, p={p_val:.6f}, {'DRIFT' if drift else 'ok'}")

drift_share = sum(v for k, v in drift_metrics.items() if k.startswith("drift_")) / 4
drift_metrics["drift_share"] = drift_share
print(f"\nDrift share: {drift_share:.0%}")

report = {"summary": {"drift_share": drift_share, "n_reference": len(ref_f), "n_current": len(cur_f)}, "features": feat_details}
Path("/opt/airflow/project/reports/drift").mkdir(parents=True, exist_ok=True)
Path("/opt/airflow/project/reports/drift/drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

rows = "\n".join(
    f'<tr class="{"drift" if v["drift_detected"] else "nodrift"}"><td>{c}</td>'
    f'<td>{v["ks_stat"]:.4f}</td><td>{v["p_value"]:.6f}</td>'
    f'<td>{"YES" if v["drift_detected"] else "NO"}</td></tr>'
    for c, v in feat_details.items()
)
html = f"""<!DOCTYPE html><html><head><title>Drift Report</title>
<style>body{{font-family:sans-serif;margin:40px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:8px}}th{{background:#4CAF50;color:white}}
.drift{{background:#ffcccc}}.nodrift{{background:#ccffcc}}</style></head><body>
<h1>Rakuten MLOps — Data Drift Report (KS-test)</h1>
<p>Reference: X_train_clean ({len(ref_f)} rows) | Current: X_val_clean ({len(cur_f)} rows)</p>
<h3>Drift Share: {drift_share:.0%}</h3>
<table><tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Drift Detected (p&lt;0.05)</th></tr>
{rows}</table></body></html>"""
Path("/opt/airflow/project/reports/drift/drift_report.html").write_text(html, encoding="utf-8")

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("rakuten-data-drift")
with mlflow.start_run(run_name="data-drift-ks-test"):
    mlflow.log_metrics(drift_metrics)
    mlflow.log_param("method", "KS-test (scipy)")
    mlflow.log_param("features", "text_length,has_description,description_length,word_count")
    try:
        mlflow.log_artifact("/opt/airflow/project/reports/drift/drift_report.json", artifact_path="drift")
        mlflow.log_artifact("/opt/airflow/project/reports/drift/drift_report.html", artifact_path="drift")
    except Exception as e:
        print(f"  Artifact logging skipped ({e}) — files saved locally in reports/drift/")
print("\nMLflow drift run logged successfully!")
