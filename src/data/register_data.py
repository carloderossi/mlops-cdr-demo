"""
register_data.py
----------------
Generates synthetic datasets locally and registers them as Azure ML Data Assets
in the workspace. Run this once before submitting any training jobs.

Usage:
    & c:/Carlo/Azure/AI-300/mlops-cdr-demo/.venv/Scripts/python.exe c:/Carlo/Azure/AI-300/mlops-cdr-demo/src/data/register_data.py
"""

import argparse
import os
import subprocess
import sys

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from src.auth import getMLClient


ASSETS = [
    {
        "name": "credit-risk-baseline-train",
        "filename": "baseline_train.parquet",
        "description": "Baseline training data — stable economy conditions (Champion ModelA)",
        "tags": {"dataset_type": "train", "regime": "stable", "project": "mlops-cdr-demo"},
    },
    {
        "name": "credit-risk-baseline-test",
        "filename": "baseline_test.parquet",
        "description": "Baseline held-out test data — stable economy conditions",
        "tags": {"dataset_type": "test", "regime": "stable", "project": "mlops-cdr-demo"},
    },
    {
        "name": "credit-risk-drifted-inference",
        "filename": "drifted_inference.parquet",
        "description": "Drifted production inference data — economic downturn, no labels (delayed ground truth)",
        "tags": {"dataset_type": "inference", "regime": "downturn", "project": "mlops-cdr-demo"},
    },
    {
        "name": "credit-risk-drifted-labels",
        "filename": "drifted_labels.parquet",
        "description": "Delayed ground truth labels for drifted inference data",
        "tags": {"dataset_type": "labels", "regime": "downturn", "project": "mlops-cdr-demo"},
    },
    {
        "name": "credit-risk-drifted-train",
        "filename": "drifted_train.parquet",
        "description": "Challenger retrain dataset — 70% stable + 100% drifted (economic downturn)",
        "tags": {"dataset_type": "train", "regime": "mixed", "project": "mlops-cdr-demo"},
    },
]


def main(args):
    # 1. Generate datasets locally
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    print(f"[INFO] Generating datasets into: {data_dir}")
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "generate_data.py"),
         "--output_dir", data_dir, "--seed", "42"],
        check=True,
    )
    print("[INFO] Dataset generation complete.")

    # 2. Connect to Azure ML workspace
    print("[INFO] Connecting to Azure ML workspace...")
    # ml_client = MLClient(
    #     credential=DefaultAzureCredential(),
    #     subscription_id=args.subscription_id,
    #     resource_group_name=args.resource_group,
    #     workspace_name=args.workspace_name,
    # )
    ml_client = getMLClient(None)

    # 3. Register each dataset as a versioned Data Asset
    for asset in ASSETS:
        local_path = os.path.join(data_dir, asset["filename"])
        if not os.path.exists(local_path):
            print(f"[WARNING] File not found, skipping: {local_path}")
            continue

        print(f"[INFO] Registering: {asset['name']} ← {local_path}")
        data_asset = Data(
            name=asset["name"],
            description=asset["description"],
            tags=asset["tags"],
            path=local_path,
            type=AssetTypes.URI_FILE,
        )
        registered = ml_client.data.create_or_update(data_asset)
        print(f"  ✓ {registered.name}  version={registered.version}")

    print("\n[OK] All data assets registered.")
    print("     Verify in Azure ML Studio → Data → Data Assets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register synthetic credit risk data in Azure ML")
    # parser.add_argument("--subscription_id", required=True)
    # parser.add_argument("--resource_group",  required=True)
    # parser.add_argument("--workspace_name",  required=True)
    parser.add_argument("--data_dir", default="./data",
                        help="Local directory for generated parquet files")
    main(parser.parse_args())
