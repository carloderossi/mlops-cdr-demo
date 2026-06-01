"""
submit_train_job.py
-------------------
Azure ML SDK v2 — Credit Risk Champion/Challenger Training Job Submitter.

Replaces train_job.yml and retrain_job.yml with a fully programmatic workflow:

  1. Build and submit a CommandJob (Champion or Challenger mode)
  2. Stream / poll job status with live progress reporting
  3. On success — register the MLflow model output in the Azure ML Registry
     with full Champion–Challenger lifecycle tags (version, status, metrics,
     drift trigger, rollout %, deployment date)

The MLflow model artifact is written by train.py into the job output.
This script collects that output path and calls ml_client.models.create_or_update
to register it — keeping train.py pure (train-only, no SDK dependency).

Usage:
    # Champion (initial training on stable baseline data):
    python jobs/submit_train_job.py \\
        --subscription_id <sub>  \\
        --resource_group  <rg>   \\
        --workspace_name  <ws>   \\
        --compute_cluster <cluster-name>

    # Challenger (retrain after drift detection):
    python jobs/submit_train_job.py \\
        --subscription_id <sub>  \\
        --resource_group  <rg>   \\
        --workspace_name  <ws>   \\
        --compute_cluster <cluster-name> \\
        --mode challenger \\
        --drift_trigger  "covariate_drift: income -17%, debt_ratio +20%"

    # Dry-run (build job object and print config, no submission):
    python jobs/submit_train_job.py ... --dry_run

Requirements (installed in your local venv, NOT needed inside AML):
    pip install azure-ai-ml azure-identity


Usage:
    & c:/Carlo/Azure/AI-300/mlops-cdr-demo/.venv/Scripts/python.exe -m src.train.submit_train_job --compute_cluster ml-ai300cdr-cluster
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Model
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

from src.auth import getMLClient

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "credit-risk-champion-challenger"
ENVIRONMENT_NAME = "credit-risk-train" #"credit-risk-train@latest"
MODEL_NAME = "credit-risk-model"

# Dataset asset names (registered by data/register_data.py)
DATASETS = {
    "champion": {
        "train": "credit-risk-baseline-train@latest",
        "test":  "credit-risk-baseline-test@latest",
    },
    "challenger": {
        "train": "credit-risk-drifted-train@latest",
        "test":  "credit-risk-baseline-test@latest",   # same test set → fair comparison
    },
}

# Hyperparameters per mode
HYPERPARAMS = {
    "champion": dict(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    ),
    "challenger": dict(
        n_estimators=250,
        max_depth=4,        # more conservative — smaller drifted sample
        learning_rate=0.04,
        subsample=0.75,
        reg_alpha=0.2,
        reg_lambda=1.5,
    ),
}

# Polling interval (seconds) when streaming is not available
POLL_INTERVAL = 10


from pathlib import Path

def get_train_env_path() -> Path:
    # File location: {proj_root}/src/train/your_script.py
    current_file = Path(__file__).resolve()
    proj_root = current_file.parents[2]   # go from train → src → proj_root
    return proj_root / "infra" / "envs" / "train-env.yml"


def create_env(ml_client):
    env_name = ENVIRONMENT_NAME
    env_version = "v1"

    try:
        path = get_train_env_path()
        print(path, path.exists())

        env = ml_client.environments.get(env_name, env_version)
        print(f"Environment already exists: {env.name}:{env.version}")
    except Exception:
        print("Creating new environment...")
        env = Environment(
            name=env_name,
            # version=env_version, # yaml file hashing comparison
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file=get_train_env_path() 
        )
        ml_client.environments.create_or_update(env)
        print(f"Environment created: {env.name} - {env.version}")

    return env

# ---------------------------------------------------------------------------
# Job builder
# ---------------------------------------------------------------------------

def build_job(args: argparse.Namespace, env: Environment) -> command:
    """Construct the Azure ML CommandJob object from parsed args."""
    mode = args.mode
    hp = HYPERPARAMS[mode]
    ds = DATASETS[mode]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"credit-risk-{mode}-train-{timestamp}"
    display_name = (
        f"Credit Risk — {'Champion Training (ModelA)' if mode == 'champion' else 'Challenger Retraining (ModelB)'}"
    )

    # Build the CLI command string — mirrors what was in the YAML
    hp_flags = " ".join(f"--{k} {v}" for k, v in hp.items())
    command_str = (
        f"python train.py"
        f" --train_data ${{{{inputs.train_data}}}}"
        f" --test_data ${{{{inputs.test_data}}}}"
        f" --model_output ${{{{outputs.model_output}}}}"
        f" --model_name {MODEL_NAME}"
        f" {hp_flags}"
    )

    job = command(
        name=job_name,
        display_name=display_name,
        description=(
            f"{'Initial Champion' if mode == 'champion' else 'Drift-triggered Challenger'} "
            f"XGBoost Credit Risk training. "
            f"Submitted: {timestamp}."
        ),
        experiment_name=EXPERIMENT_NAME,
        tags={
            "project":      "mlops-cdr-demo",
            "mode":         mode,
            "model_role":   mode,
            "dataset":      "baseline" if mode == "champion" else "drifted_mix",
            "algorithm":    "xgboost",
            "author":       "carlo",
            "submitted_at": timestamp,
        },
        # ── Code: repo root (src/ must be importable) ──────────────────────
        code="./src/train", #str(Path(__file__).parent.parent),   # project root
        command=command_str,
        environment=env, #f"azureml:{ENVIRONMENT_NAME}",
        compute=f"{args.compute_cluster}", #f"azureml:{args.compute_cluster}",
        # ── Inputs ──────────────────────────────────────────────────────────
        inputs={
            "train_data": Input(
                type=AssetTypes.URI_FILE,
                path=ds['train'], #f"azureml:{ds['train']}",
                mode=InputOutputModes.RO_MOUNT,
            ),
            "test_data": Input(
                type=AssetTypes.URI_FILE,
                path=ds['test'], #f"azureml:{ds['test']}",
                mode=InputOutputModes.RO_MOUNT,
            ),
        },
        # ── Outputs ─────────────────────────────────────────────────────────
        outputs={
            "model_output": Output(
                type=AssetTypes.MLFLOW_MODEL,
                mode=InputOutputModes.RW_MOUNT,
            ),
        },
        # ── Resources ───────────────────────────────────────────────────────
        instance_count=1,
    )

    return job


# ---------------------------------------------------------------------------
# Job submission & polling
# ---------------------------------------------------------------------------

TERMINAL_STATES = {"Completed", "Failed", "Canceled", "NotResponding"}


def submit_and_wait(ml_client: MLClient, job) -> tuple[str, str]:
    """
    Submit a job and block until it reaches a terminal state.
    Returns (job_name, final_status).
    """
    log.info("Submitting job: %s", job.name)
    submitted = ml_client.jobs.create_or_update(job)
    job_name = submitted.name
    print("Submitted job:", job_name)

    studio_url = getattr(submitted, "studio_url", None)
    if studio_url:
        log.info("Studio URL: %s", studio_url)

    log.info("Polling every %ds — waiting for terminal state...", POLL_INTERVAL)
    last_status = None

    while True:
        time.sleep(POLL_INTERVAL)
        current = ml_client.jobs.get(job_name)
        status = current.status
        print(f"Job status: {status}")
        if status != last_status:
            log.info("  Job %-40s  status: %s", job_name, status)
            last_status = status

        if status in TERMINAL_STATES:
            break

    log.info("Job finished with status: %s", last_status)
    return job_name, last_status


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------

def read_eval_metrics(ml_client: MLClient, job_name: str) -> dict:
    """
    Attempt to read eval_metrics.json from the job output.
    Falls back gracefully — registration proceeds even without metrics.
    """
    try:
        # Download the model output artifact to a temp location
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ml_client.jobs.download(
                name=job_name,
                output_name="model_output",
                download_path=tmp,
            )
            metrics_path = Path(tmp) / "named-outputs" / "model_output" / "eval_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                log.info("Eval metrics loaded: %s", json.dumps(metrics))
                return metrics
    except Exception as exc:
        log.warning("Could not read eval_metrics.json: %s", exc)
    return {}


def register_model(
    ml_client: MLClient,
    job_name: str,
    mode: str,
    drift_trigger: str,
    initial_traffic_pct: int,
    metrics: dict,
) -> Model:
    """
    Register the MLflow model artifact from the completed job into the
    Azure ML Registry with full Champion–Challenger lifecycle tags.
    """
    deployment_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "champion" if mode == "champion" else "challenger_candidate"

    tags = {
        # Lifecycle identity
        "champion_challenger_status": status,
        "model_role":                 mode,
        "deployment_date":            deployment_date,
        "training_job":               job_name,
        # Evaluation
        "roc_auc":          str(metrics.get("roc_auc", "n/a")),
        "f1_score":         str(metrics.get("f1_score", "n/a")),
        "precision":        str(metrics.get("precision", "n/a")),
        "recall":           str(metrics.get("recall", "n/a")),
        # Operational
        "drift_trigger":    drift_trigger or ("initial_training" if mode == "champion" else "covariate_drift"),
        "traffic_pct":      str(initial_traffic_pct),
        "rollback_model":   "n/a",          # filled in by deployment script at promotion time
        "project":          "mlops-cdr-demo",
    }

    log.info("Registering model '%s' from job output '%s'...", MODEL_NAME, job_name)
    log.info("  Tags: %s", json.dumps(tags, indent=4))

    model = Model(
        # Reference the job output directly — AML resolves the artifact URI
        path=f"azureml://jobs/{job_name}/outputs/model_output",
        name=MODEL_NAME,
        description=(
            f"XGBoost Credit Risk classifier — "
            f"{'initial Champion (stable baseline data)' if mode == 'champion' else 'Challenger candidate (drift-retrained)'}"
        ),
        type=AssetTypes.MLFLOW_MODEL,
        tags=tags,
    )

    registered = ml_client.models.create_or_update(model)

    log.info(
        "Model registered: name=%s  version=%s  status=%s  roc_auc=%s",
        registered.name,
        registered.version,
        status,
        metrics.get("roc_auc", "n/a"),
    )
    return registered


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(job_name: str, job_status: str, model: Model | None, args: argparse.Namespace):
    width = 64
    bar = "─" * width
    print(f"\n┌{bar}┐")
    print(f"│{'  Credit Risk MLOps — Job Summary':^{width}}│")
    print(f"├{bar}┤")
    rows = [
        ("Job name",       job_name),
        ("Mode",           args.mode),
        ("Final status",   job_status),
        ("Compute",        args.compute_cluster),
        ("Experiment",     EXPERIMENT_NAME),
    ]
    if model:
        rows += [
            ("Model name",    model.name),
            ("Model version", model.version),
            ("Model status",  model.tags.get("champion_challenger_status", "n/a")),
            ("ROC-AUC",       model.tags.get("roc_auc", "n/a")),
            ("Traffic %",     model.tags.get("traffic_pct", "n/a")),
            ("Drift trigger", model.tags.get("drift_trigger", "n/a")),
        ]
    for label, value in rows:
        print(f"│  {label:<22}{str(value):<{width - 24}}│")
    print(f"└{bar}┘\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit Azure ML Credit Risk training job and register the model."
    )
    # Workspace identity
    parser.add_argument("--compute_cluster", required=True,
                        help="Name of the AML compute cluster (without 'azureml:' prefix)")

    # Job mode
    parser.add_argument("--mode", choices=["champion", "challenger"], default="champion",
                        help="champion = initial baseline training; challenger = drift-retrain")

    # Challenger-specific
    parser.add_argument("--drift_trigger", type=str, default=None,
                        help="Human-readable drift description stored as model tag "
                             "(e.g. 'covariate_drift: income -17%%')")
    parser.add_argument("--initial_traffic_pct", type=int, default=None,
                        help="Initial traffic allocation. Defaults: 100 for champion, 0 for challenger.")

    # Behaviour flags
    parser.add_argument("--skip_registration", action="store_true",
                        help="Submit and wait but do NOT register the model (useful for debugging)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build the job object and print config without submitting")

    return parser.parse_args()


def main():
    args = parse_args()

    # Default traffic percentages
    if args.initial_traffic_pct is None:
        args.initial_traffic_pct = 100 if args.mode == "champion" else 0

    # ── Connect ─────────────────────────────────────────────────────────────
    log.info("Connecting to workspace 'mlw-ai300-cdrlabs' ...")
    ml_client = getMLClient(None)

    # ── Environment ────────────────────────────────────────────────────────────
    env = create_env(ml_client)

    # ── Build job ────────────────────────────────────────────────────────────
    job = build_job(args, env)

    print(f"Validating compute '{job.compute}'")
    ml_client.compute.get(job.compute)

    if args.dry_run:
        log.info("DRY RUN — job configuration:")
        log.info("  name:        %s", job.name)
        log.info("  display:     %s", job.display_name)
        log.info("  experiment:  %s", job.experiment_name)
        log.info("  environment: %s", job.environment)
        log.info("  compute:     %s", job.compute)
        log.info("  command:     %s", job.command)
        log.info("  inputs:      %s", {k: v.path for k, v in job.inputs.items()})
        log.info("  tags:        %s", job.tags)
        log.info("Dry run complete — no job submitted.")
        return

    # ── Submit + poll ────────────────────────────────────────────────────────
    job_name, job_status = submit_and_wait(ml_client, job)

    if job_status != "Completed":
        log.error("Job did not complete successfully (status=%s). Skipping registration.", job_status)
        print_summary(job_name, job_status, None, args)
        sys.exit(1)

    # ── Read metrics from job artifact ───────────────────────────────────────
    metrics = {}
    if not args.skip_registration:
        metrics = read_eval_metrics(ml_client, job_name)

    # ── Register model ───────────────────────────────────────────────────────
    registered_model = None
    if not args.skip_registration:
        registered_model = register_model(
            ml_client=ml_client,
            job_name=job_name,
            mode=args.mode,
            drift_trigger=args.drift_trigger,
            initial_traffic_pct=args.initial_traffic_pct,
            metrics=metrics,
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(job_name, job_status, registered_model, args)


if __name__ == "__main__":
    main()
