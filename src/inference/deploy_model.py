import time
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential

from src.auth import getMLClient

"""
    Usage:
        & c:/Carlo/Azure/AI-300/mlops-cdr-demo/.venv/Scripts/python.exe -m src.inference.deploy_model
"""

# 1. Connect to your Azure ML Workspace using the provided details
print("Authenticating and connecting to the Azure ML workspace...")
ml_client = getMLClient(None)

# 2. Define a unique Online Endpoint name 
# (Must be unique across the entire Azure region, e.g., westeurope)
import random
import string

# Creating a unique endpoint name by including a random suffix
allowed_chars = string.ascii_lowercase + string.digits
endpoint_suffix = "".join(random.choice(allowed_chars) for x in range(5))
endpoint_name = "credit-risk-endpoint-" + endpoint_suffix

print(f"Endpoint name: {endpoint_name}")
#endpoint_name = f"credit-risk-endpoint-{int(time.time())}"

# https://learn.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list?view=azureml-api-2
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Online endpoint for XGBoost Credit Risk classification",
    auth_mode="key",
    location="swedencentral"
)

print(f"Creating online endpoint: {endpoint_name}...")
# This registers the endpoint container in your workspace
ml_client.begin_create_or_update(endpoint).result()


# 3. Define the deployment configuration
# Note: For MLflow models, you do not need to provide a scoring_script or environment.
deployment_name = "champion-deployment"

# We reference your model asset ID explicitly
# https://github.com/Azure/azureml-examples/blob/main/sdk/python/endpoints/online/mlflow/online-endpoints-deploy-mlflow-model-with-script.ipynb
model_asset = ml_client.models.get(name="credit-risk-model", version="1")

# https://learn.microsoft.com/en-us/azure/machine-learning/concept-online-deployment-model-specification?view=azureml-api-2
blue_deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=model_asset, # model_asset_id,
    instance_type="Standard_DS2_v2",  # Choose a compute size adequate for XGBoost
    instance_count=1,
    
    # Force Azure to use a stable, pre-built XGBoost/Scikit-Learn runtime environment
    environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1"    
)

print(f"Deploying model version 1 to {deployment_name} (this may take a few minutes)...")
ml_client.online_deployments.begin_create_or_update(blue_deployment).result()


# 4. Route 100% of the traffic to this new deployment
endpoint.traffic = {deployment_name: 100}
print("Routing 100% of traffic to the new deployment...")
ml_client.begin_create_or_update(endpoint).result()

print(f"\nSuccessfully deployed! Your endpoint '{endpoint_name}' is ready to serve traffic.")