"""
train.py
--------
Azure ML training script for the Credit Risk Champion–Challenger demo.

Trains an XGBoost binary classifier on synthetic credit data.
Logs all metrics, parameters, and artifacts to MLflow.
Registers the model in the Azure ML Registry using MLflow format.

Inputs  (Azure ML job input paths):
    --train_data    : path to training parquet file
    --test_data     : path to test parquet file

Outputs (Azure ML job output paths):
    --model_output  : directory where the MLflow model artifact is written

Optional:
    --model_name    : name for registry registration (default: credit-risk-model)
    --n_estimators  : XGBoost n_estimators (default: 200)
    --max_depth     : XGBoost max_depth (default: 5)
    --learning_rate : XGBoost learning_rate (default: 0.05)
    --subsample     : XGBoost subsample (default: 0.8)
    --reg_alpha     : XGBoost L1 regularisation (default: 0.1)
    --reg_lambda    : XGBoost L2 regularisation (default: 1.0)
    --scale_pos_weight : XGBoost class imbalance weight (default: auto)
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
TARGET_COL = "default"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load parquet from a path that may be a file or a directory."""
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(".parquet")]
        if not files:
            raise FileNotFoundError(f"No parquet files found in directory: {path}")
        path = os.path.join(path, files[0])
        log.info("Loading from directory: %s", path)
    df = pd.read_parquet(path)
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def plot_roc_curve(fpr, tpr, auc_score: float, title: str = "ROC Curve") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(model: XGBClassifier, feature_names: list) -> plt.Figure:
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importance[sorted_idx],
        color="#2ca02c",
        alpha=0.8,
    )
    ax.set_xlabel("Gain Importance")
    ax.set_title("XGBoost Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shap_summary(model: XGBClassifier, X: pd.DataFrame) -> plt.Figure:
    log.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.tight_layout()
    return plt.gcf()


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------

def train(args):
    # --- Load data ---
    train_df = load_data(args.train_data)
    test_df = load_data(args.test_data)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    log.info("Train shape: %s | default rate: %.2f%%", X_train.shape, y_train.mean() * 100)
    log.info("Test  shape: %s | default rate: %.2f%%", X_test.shape, y_test.mean() * 100)

    # --- Class imbalance weight ---
    if args.scale_pos_weight is None:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos
        log.info("Auto scale_pos_weight: %.2f (neg=%d, pos=%d)", scale_pos_weight, neg, pos)
    else:
        scale_pos_weight = args.scale_pos_weight

    # --- MLflow run ---
    mlflow.xgboost.autolog(log_models=False)  # we log manually for full control

    with mlflow.start_run():

        # Log parameters
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
        }
        mlflow.log_params(params)

        # --- Train ---
        log.info("Training XGBoost classifier...")
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=0.8,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=5,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # --- Evaluate ---
        log.info("Evaluating model...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        metrics = {
            "roc_auc": round(roc_auc, 6),
            "f1_score": round(f1, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "avg_precision_score": round(avg_precision, 6),
            "best_iteration": model.best_iteration,
        }
        mlflow.log_metrics(metrics)
        log.info("Metrics: %s", json.dumps(metrics, indent=2))

        # Classification report
        report = classification_report(y_test, y_pred, target_names=["no_default", "default"])
        log.info("Classification report:\n%s", report)
        mlflow.log_text(report, "classification_report.txt")

        # --- Artifacts: plots ---
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
        mlflow.log_figure(roc_fig, "roc_curve.png")
        plt.close(roc_fig)

        fi_fig = plot_feature_importance(model, FEATURE_COLS)
        mlflow.log_figure(fi_fig, "feature_importance.png")
        plt.close(fi_fig)

        try:
            shap_fig = plot_shap_summary(model, X_test)
            mlflow.log_figure(shap_fig, "shap_summary.png")
            plt.close(shap_fig)
        except Exception as exc:
            log.warning("SHAP plot skipped: %s", exc)

        # --- Feature importance as JSON (for dashboard) ---
        importance_dict = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
        mlflow.log_dict(importance_dict, "feature_importance.json")

        # --- Save model (MLflow format) ---
        log.info("Saving model to: %s", args.model_output)
        os.makedirs(args.model_output, exist_ok=True)

        # Signature for schema validation at inference time
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])

        mlflow.xgboost.save_model(
            model,
            path=args.model_output,
            signature=signature,
            input_example=X_train.head(5),
        )
        log.info("Model artifact saved.")

        # Save metrics JSON alongside model for downstream jobs
        metrics_path = os.path.join(args.model_output, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Metrics written to %s", metrics_path)

        log.info("Training complete. ROC-AUC = %.4f", roc_auc)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Credit Risk XGBoost Training Job")

    # Data paths (provided by Azure ML job inputs)
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training parquet (file or directory)")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test parquet (file or directory)")

    # Model output path (provided by Azure ML job output)
    parser.add_argument("--model_output", type=str, required=True,
                        help="Directory to write the MLflow model artifact")

    # Model identity
    parser.add_argument("--model_name", type=str, default="credit-risk-model",
                        help="Model name for MLflow / registry logging")

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--reg_alpha", type=float, default=0.1)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--scale_pos_weight", type=float, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Log model name as MLflow tag
    mlflow.set_tag("model_name", args.model_name)
    mlflow.set_tag("champion_challenger_status", "candidate")
    train(args)
