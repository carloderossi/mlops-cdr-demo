"""
inference_client.py
-------------------
Sends inference requests to the Azure ML Managed Online Endpoint
and writes a timestamped inference log for downstream drift detection.

Modes
-----
  --mode endpoint   : calls the real AML REST endpoint (production)
  --mode local      : calls a locally-served MLflow model (dev/test, no Azure needed)

The inference log written by this script is what detect_drift.py reads.
It contains the original feature values + the predicted probability,
so the drift detector can compare the feature distribution against baseline.

Usage — endpoint mode:
    python src/inference_client.py \\
        --mode endpoint \\
        --endpoint_url  https://<endpoint>.<region>.inference.ml.azure.com/score \\
        --api_key       <key-or-use-env-var ENDPOINT_API_KEY> \\
        --input_data    ./data/drifted_inference.parquet \\
        --output_log    ./data/inference_log.parquet \\
        --batch_size    50

Usage — local mode (mlflow model served with `mlflow models serve`):
    mlflow models serve -m ./outputs/credit_risk_model -p 5001 --no-conda
    python src/inference_client.py \\
        --mode local \\
        --input_data ./data/drifted_inference.parquet \\
        --output_log ./data/inference_log.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

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

# Columns to cast to float64 to match the MLflow signature schema
INT_COLS = ["age", "credit_score", "num_credit_lines", "num_late_payments"]

LOCAL_ENDPOINT = "http://127.0.0.1:5001/invocations"
ENDPOINT="https://mlw-ai300-cdrlabs-vwhit.westeurope.inference.ml.azure.com/score"


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def build_aml_payload(batch: pd.DataFrame) -> dict:
    """
    AML Online Endpoint expects the MLflow scoring schema:
    { "input_data": { "columns": [...], "data": [[...], ...] } }
    """
    return {
        "input_data": {
            "columns": FEATURE_COLS,
            "data": batch[FEATURE_COLS].values.tolist(),
        }
    }


def build_local_payload(batch: pd.DataFrame) -> dict:
    """
    Local mlflow models serve expects:
    { "dataframe_records": [{col: val, ...}, ...] }
    """
    return {"dataframe_records": batch[FEATURE_COLS].to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Single-batch sender
# ---------------------------------------------------------------------------

def send_batch(
    batch: pd.DataFrame,
    mode: str,
    endpoint_url: str,
    api_key: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> list[float]:
    """
    Send one batch to the endpoint. Returns list of predicted probabilities.
    Retries with exponential backoff on transient errors.
    """
    if mode == "endpoint":
        payload = build_aml_payload(batch)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        url = endpoint_url
    else:
        payload = build_local_payload(batch)
        headers = {"Content-Type": "application/json"}
        url = ENDPOINT #LOCAL_ENDPOINT

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            # AML endpoint returns {"predictions": [...]} or a raw list
            if isinstance(result, dict) and "predictions" in result:
                return result["predictions"]
            elif isinstance(result, list):
                return result
            else:
                # MLflow local server may wrap differently
                return list(result)

        except requests.exceptions.RequestException as exc:
            if attempt == retries:
                raise
            wait = backoff ** attempt
            log.warning("Attempt %d/%d failed (%s). Retrying in %.1fs...", attempt, retries, exc, wait)
            time.sleep(wait)

    return []   # unreachable


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> pd.DataFrame:
    # --- Load input data ---
    log.info("Loading inference data from: %s", args.input_data)
    df = pd.read_parquet(args.input_data)

    # Cast int cols to float64 to match trained model signature
    df = df.copy()
    for col in INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    log.info("Loaded %d rows | columns: %s", len(df), list(df.columns))

    api_key = args.api_key or os.environ.get("ENDPOINT_API_KEY", "")
    if args.mode == "endpoint" and not api_key:
        raise ValueError("--api_key or env var ENDPOINT_API_KEY is required in endpoint mode")

    # --- Send batches ---
    n_rows = len(df)
    all_predictions: list[float] = []
    all_timestamps: list[str] = []

    batch_size = args.batch_size
    n_batches = (n_rows + batch_size - 1) // batch_size

    log.info("Sending %d rows in %d batches of %d to [%s]...",
             n_rows, n_batches, batch_size, args.mode)

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_rows)
        batch = df.iloc[start_idx:end_idx]

        t0 = time.time()
        preds = send_batch(
            batch=batch,
            mode=args.mode,
            endpoint_url=args.endpoint_url,
            api_key=api_key,
        )
        latency_ms = (time.time() - t0) * 1000

        timestamp = datetime.now(timezone.utc).isoformat()
        all_predictions.extend(preds)
        all_timestamps.extend([timestamp] * len(preds))

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            log.info(
                "  Batch %d/%d | rows %d–%d | latency %.0fms | avg_pred_prob %.3f",
                i + 1, n_batches, start_idx, end_idx,
                latency_ms, float(np.mean(preds)) if preds else 0.0,
            )

        # Small sleep to avoid hammering the endpoint in demo
        if args.mode == "endpoint":
            time.sleep(args.request_delay)

    # --- Assemble inference log ---
    log_df = df[FEATURE_COLS].copy().iloc[:len(all_predictions)].reset_index(drop=True)
    log_df["predicted_prob"] = all_predictions
    log_df["predicted_label"] = (log_df["predicted_prob"] >= 0.5).astype(int)
    log_df["inference_timestamp"] = all_timestamps
    log_df["model_role"] = args.model_role
    log_df["endpoint_mode"] = args.mode

    # --- Summary stats ---
    log.info(
        "Inference complete | rows=%d | predicted_default_rate=%.2f%%",
        len(log_df),
        log_df["predicted_label"].mean() * 100,
    )

    # --- Write inference log ---
    Path(args.output_log).parent.mkdir(parents=True, exist_ok=True)
    log_df.to_parquet(args.output_log, index=False)
    log.info("Inference log written to: %s", args.output_log)

    return log_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit Risk Inference Client")

    parser.add_argument("--mode", choices=["endpoint", "local"], default="local",
                        help="'endpoint' = AML Managed Online Endpoint; 'local' = mlflow models serve")

    # Endpoint
    parser.add_argument("--endpoint_url", type=str, default="",
                        help="AML endpoint scoring URI (endpoint mode only)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (or set env var ENDPOINT_API_KEY)")

    # Data
    parser.add_argument("--input_data", type=str, default="./data/drifted_inference.parquet",
                        help="Path to input parquet (features, no labels)")
    parser.add_argument("--output_log", type=str, default="./data/inference_log.parquet",
                        help="Path where the inference log parquet is written")

    # Batching
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Number of rows per HTTP request")
    parser.add_argument("--request_delay", type=float, default=0.1,
                        help="Sleep (seconds) between batches in endpoint mode")

    # Metadata tag written to inference log
    parser.add_argument("--model_role", type=str, default="champion",
                        help="Model role tag stored in inference log (champion/challenger)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
