from datetime import datetime

from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    CredentialUnavailableError,
    CertificateCredential
)
from azure.core.exceptions import AzureError
from azure.ai.ml import MLClient

import json
from pathlib import Path
from dotenv import load_dotenv
import os

# ANSI colour codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

def dump_user(credentials):
    import requests
    from azure.identity import DefaultAzureCredential
    # Request Graph token
    token = credentials.get_token("https://graph.microsoft.com/.default")

    headers = {
        "Authorization": f"Bearer {token.token}",
        "Content-Type": "application/json"
    }

    # Call Microsoft Graph
    response = requests.get(
        "https://graph.microsoft.com/v1.0/me",
        headers=headers
    )

    user = response.json()
    print(f"{CYAN}User display name:{RESET} {GREEN}{user.get('displayName')}{RESET}")
    print(f"{CYAN}User object ID:{RESET} {YELLOW}{user.get('id')}{RESET}")
    print(f"{CYAN}User email:{RESET} {MAGENTA}{user.get('mail') or user.get('userPrincipalName')}{RESET}")

def get_credentials(config_path) -> CertificateCredential:
    """
    Returns a valid Azure credential for the current environment.
    """
    print(os.getcwd())

    load_dotenv(override=True, dotenv_path="./.azure/mlops-cdr-demo/.env")
    TENANT_ID = os.getenv("AZURE_TENANT_ID")
    print(f"{CYAN}using TENANT_ID: '{RESET} {YELLOW}{TENANT_ID}'")
    SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
    print(f"{CYAN}using SUBSCRIPTION_ID: '{RESET} {YELLOW}{SUBSCRIPTION_ID}'")
    CLIENT_ID = os.getenv("AZURE_CLIENT_ID") # The ID for 'cdr-foundry-client'
    print(f"{CYAN}using CLIENT_ID: '{RESET} {YELLOW}{CLIENT_ID}'")
    print(f"{RESET}")

    # @TODO: add to .azure/{proj}/.env
    # CERT_PATH = Path(r"C:\Carlo\Azure\certAuth\local\certs\azure\clientcert_token.pem")
    CERT_PATH = Path(r"C:\Carlo\Azure\certAuth\local\certs\azure\infracert_token.pem")
    print(f"using certificate from: '{CERT_PATH}'")

    try:
        # credential = DefaultAzureCredential()
        credential = CertificateCredential(
            tenant_id=TENANT_ID,
            client_id=CLIENT_ID,
            certificate_path=CERT_PATH
        )
        # Check if given credential can get token successfully.
        token = credential.get_token("https://management.azure.com/.default")
        print("Successfully obtained credentials using DefaultAzureCredential.")
        expiry = datetime.fromtimestamp(token.expires_on)
        print("Access token expires at:", expiry)
        return credential
    except Exception as ex:
        print("Failed to obtain Client Cert Credentials.", ex)
        raise RuntimeError("Failed to obtain Azure credentials: Please check your configuration and environment.")

def getMLClient(config_path: str):
    credentials = get_credentials(None)
    SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
    print(f"{CYAN}Using subscription:{RESET} {YELLOW}{SUBSCRIPTION_ID}{RESET}")
    WORKSPACE_NAME = os.environ["AZURE_ML_WORKSPACE"]
    print(f"{CYAN}Using workspace:{RESET} {GREEN}{WORKSPACE_NAME}{RESET}")
    RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP"]
    print(f"{CYAN}Using resource group:{RESET} {YELLOW}{RESOURCE_GROUP}{RESET}")

    ml_client = MLClient(
        credentials,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    return ml_client

if __name__ == "__main__":
    mlclient = getMLClient(None)
    print(f"{CYAN}Successfully obtained credentials:{RESET} {BLUE}", mlclient)
