"""
detect_drift.py
---------------
Detects data drift between the baseline training distribution and
the current inference log produced by inference_client.py.

Drift metrics
-------------
  PSI  (Population Stability Index) — primary signal, per feature
       Standard credit risk model monitoring metric.
       Threshold 0.25 → retraining trigger.

  KS   (Kolmogorov–Smirnov test) — secondary signal, per feature
       p-value < 0.05 and D-statistic > 0.1 → flagged.

  Delta % — human-readable mean shift per feature.
       Used in the Streamlit drift explanation panel.

Exit codes
----------
  0 — no significant drift detected
  1 — drift threshold exceeded (GitHub Actions reads this to gate retraining)

Outputs
-------
  drift_report.json  — full per-feature stats + overall verdict + summary text
                       consumed by the dashboard and GitHub Actions workflow

Usage:
    python src/detect_drift.py \\
        --baseline_data  ./data/baseline_train.parquet \\
        --inference_log  ./data/inference_log.parquet  \\
        --output_report  ./data/drift_report.json      \\
        --psi_threshold  0.25

    # In CI — non-zero exit triggers the retraining workflow:
    python src/detect_drift.py ... || echo "DRIFT_DETECTED=true" >> $GITHUB_OUTPUT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "income",
    "age",
    "debt_ratio",
    "num_credit_lines",
    "employment_years",
    "credit_score",
    "num_late_payments",
    "loan_amount",
    "loan_to_income",
]

PSI_BINS = 10          # number of quantile bins for PSI calculation
PSI_EPSILON = 1e-6     # avoid log(0)


# ---------------------------------------------------------------------------
# PSI calculation
# ---------------------------------------------------------------------------

def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = PSI_BINS,
) -> float:
    """
    Population Stability Index.

    PSI = Σ (current_pct - baseline_pct) * ln(current_pct / baseline_pct)

    Bins are defined on the baseline quantiles so the reference distribution
    always has uniform bin occupancy.
    """
    # Define bin edges on baseline quantiles (avoids empty baseline bins)
    breakpoints = np.quantile(baseline, np.linspace(0, 1, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts  = np.histogram(current,  bins=breakpoints)[0]

    baseline_pct = baseline_counts / len(baseline) + PSI_EPSILON
    current_pct  = current_counts  / len(current)  + PSI_EPSILON

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def psi_severity(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    elif psi < 0.25:
        return "moderate"
    else:
        return "significant"


# ---------------------------------------------------------------------------
# Per-feature drift stats
# ---------------------------------------------------------------------------

def compute_feature_drift(
    baseline: pd.Series,
    current: pd.Series,
    feature_name: str,
) -> dict:
    psi = compute_psi(baseline.dropna().values, current.dropna().values)
    ks_stat, ks_pval = stats.ks_2samp(baseline.dropna().values, current.dropna().values)

    b_mean = float(baseline.mean())
    c_mean = float(current.mean())
    delta_pct = ((c_mean - b_mean) / abs(b_mean)) * 100 if b_mean != 0 else 0.0

    b_std = float(baseline.std())
    c_std = float(current.std())

    ks_flagged = bool(ks_pval < 0.05 and ks_stat > 0.10)

    return {
        "feature":        feature_name,
        "psi":            round(psi, 6),
        "psi_severity":   psi_severity(psi),
        "ks_statistic":   round(float(ks_stat), 6),
        "ks_pvalue":      round(float(ks_pval), 6),
        "ks_flagged":     ks_flagged,
        "baseline_mean":  round(b_mean, 4),
        "current_mean":   round(c_mean, 4),
        "delta_pct":      round(delta_pct, 2),
        "baseline_std":   round(b_std, 4),
        "current_std":    round(c_std, 4),
    }


# ---------------------------------------------------------------------------
# Drift explanation summary  (for Streamlit panel)
# ---------------------------------------------------------------------------

def build_explanation_summary(feature_stats: list[dict], psi_threshold: float) -> dict:
    """
    Builds the drift explanation text matching the spec:
      "average income decreased by 18%, debt ratio increased by 22%, ..."
    """
    drifted = [f for f in feature_stats if f["psi"] >= psi_threshold]
    moderate = [f for f in feature_stats if 0.10 <= f["psi"] < psi_threshold]
    stable = [f for f in feature_stats if f["psi"] < 0.10]

    lines = []
    for f in sorted(drifted + moderate, key=lambda x: abs(x["delta_pct"]), reverse=True):
        direction = "increased" if f["delta_pct"] > 0 else "decreased"
        lines.append(
            f"• {f['feature'].replace('_', ' ')} {direction} by {abs(f['delta_pct']):.1f}%"
            f"  (PSI={f['psi']:.3f}, {f['psi_severity']})"
        )

    return {
        "n_features_significant": len(drifted),
        "n_features_moderate":    len(moderate),
        "n_features_stable":      len(stable),
        "explanation_lines":      lines,
        "summary_text": (
            f"{len(drifted)} feature(s) show significant drift (PSI ≥ {psi_threshold}). "
            + ("; ".join(
                f"{f['feature'].replace('_', ' ')} "
                f"{'↓' if f['delta_pct'] < 0 else '↑'}"
                f"{abs(f['delta_pct']):.1f}%"
                for f in sorted(drifted, key=lambda x: abs(x["psi"]), reverse=True)[:3]
            ) if drifted else "No features exceeded threshold.")
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect_drift(args: argparse.Namespace) -> dict:
    # --- Load data ---
    log.info("Loading baseline data from: %s", args.baseline_data)
    baseline_df = pd.read_parquet(args.baseline_data)

    log.info("Loading inference log from: %s", args.inference_log)
    inference_df = pd.read_parquet(args.inference_log)

    log.info("Baseline rows: %d | Inference rows: %d", len(baseline_df), len(inference_df))

    # Validate features
    missing = [c for c in FEATURE_COLS if c not in baseline_df.columns]
    if missing:
        raise ValueError(f"Missing features in baseline: {missing}")
    missing = [c for c in FEATURE_COLS if c not in inference_df.columns]
    if missing:
        raise ValueError(f"Missing features in inference log: {missing}")

    # --- Per-feature drift ---
    log.info("Computing drift statistics...")
    feature_stats = []
    for feat in FEATURE_COLS:
        stats_row = compute_feature_drift(
            baseline=baseline_df[feat],
            current=inference_df[feat],
            feature_name=feat,
        )
        feature_stats.append(stats_row)
        log.info(
            "  %-22s  PSI=%.4f (%s)  KS=%.4f (p=%.4f)  Δ=%+.1f%%",
            feat,
            stats_row["psi"], stats_row["psi_severity"],
            stats_row["ks_statistic"], stats_row["ks_pvalue"],
            stats_row["delta_pct"],
        )

    # --- Overall verdict ---
    max_psi = max(f["psi"] for f in feature_stats)
    n_significant = sum(1 for f in feature_stats if f["psi"] >= args.psi_threshold)
    drift_detected = n_significant > 0

    overall_severity = "stable"
    if drift_detected:
        overall_severity = "significant"
    elif any(f["psi_severity"] == "moderate" for f in feature_stats):
        overall_severity = "moderate"

    # --- Prediction distribution shift (if baseline predictions available) ---
    pred_shift = None
    if "predicted_prob" in inference_df.columns and "predicted_prob" in baseline_df.columns:
        pred_shift = {
            "baseline_mean_prob": round(float(baseline_df["predicted_prob"].mean()), 4),
            "current_mean_prob":  round(float(inference_df["predicted_prob"].mean()), 4),
        }
    elif "predicted_prob" in inference_df.columns:
        pred_shift = {
            "current_mean_prob": round(float(inference_df["predicted_prob"].mean()), 4),
        }

    # --- Explanation summary ---
    explanation = build_explanation_summary(feature_stats, args.psi_threshold)

    # --- Assemble report ---
    report = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "baseline_rows":       len(baseline_df),
        "inference_rows":      len(inference_df),
        "psi_threshold":       args.psi_threshold,
        "drift_detected":      drift_detected,
        "overall_severity":    overall_severity,
        "max_psi":             round(max_psi, 6),
        "n_features_flagged":  n_significant,
        "retraining_trigger":  drift_detected,
        "prediction_shift":    pred_shift,
        "explanation":         explanation,
        "feature_stats":       feature_stats,
    }

    # --- Write report ---
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Drift report written to: %s", args.output_report)

    # --- Console summary ---
    print("\n" + "─" * 60)
    print(f"  DRIFT DETECTION REPORT")
    print(f"  Generated:       {report['generated_at']}")
    print(f"  Baseline rows:   {report['baseline_rows']}")
    print(f"  Inference rows:  {report['inference_rows']}")
    print(f"  PSI threshold:   {report['psi_threshold']}")
    print(f"  Max PSI:         {report['max_psi']:.4f}")
    print(f"  Features flagged:{report['n_features_flagged']}")
    print(f"  Overall:         {overall_severity.upper()}")
    print(f"  Retrain trigger: {'YES ⚠️' if drift_detected else 'no'}")
    print("─" * 60)
    for line in explanation["explanation_lines"]:
        print(f"  {line}")
    print("─" * 60 + "\n")

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit Risk Data Drift Detector")

    parser.add_argument("--baseline_data", type=str,
                        default="./data/baseline_train.parquet",
                        help="Reference distribution — baseline training parquet")
    parser.add_argument("--inference_log", type=str,
                        default="./data/inference_log.parquet",
                        help="Current distribution — inference log parquet from inference_client.py")
    parser.add_argument("--output_report", type=str,
                        default="./data/drift_report.json",
                        help="Path for the drift report JSON")
    parser.add_argument("--psi_threshold", type=float, default=0.25,
                        help="PSI threshold above which drift triggers retraining (default: 0.25)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = detect_drift(args)

    # Non-zero exit code if retraining should be triggered.
    # GitHub Actions step reads this: `if: failure()` or `|| echo "DRIFT=true"`
    if report["retraining_trigger"]:
        log.warning("Drift threshold exceeded — exiting with code 1 to trigger retraining.")
        sys.exit(1)

    log.info("No significant drift detected — exiting with code 0.")
    sys.exit(0)
