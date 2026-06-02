"""
Azure Authentication Module

This module provides authentication functionality for Azure services, specifically
for Azure Machine Learning resources. It supports certificate-based authentication
using service principals, which is a secure method for automated/CI-CD scenarios.

Key Features:
- Certificate-based authentication using Azure service principals
- Token acquisition for Azure Resource Manager and Microsoft Graph
- MLClient initialization for Azure Machine Learning operations
- Colored console output for better user experience

Usage:
    from auth import get_credentials, getMLClient, dump_user
    
    # Get Azure credentials
    credentials = get_credentials(config_path)
    
    # Create ML client
    ml_client = getMLClient(config_path)
    
    # Display current user info
    dump_user(credentials)
"""

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

# ANSI colour codes for terminal output formatting
# These codes allow for colored text output in supported terminals
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"  # Reset color to default


def dump_user(credentials):
    """
    Retrieve and display information about the currently authenticated user.
    
    This function uses the provided credentials to obtain a token for Microsoft Graph
    and then queries the /me endpoint to retrieve user profile information.
    
    Args:
        credentials: An Azure credential object that supports get_token() method.
                    Typically a CertificateCredential or DefaultAzureCredential instance.
    
    Displays:
        - User's display name
        - User's object ID (from Entra ID/Azure AD)
        - User's email address (prefers 'mail' field, falls back to 'userPrincipalName')
    
    Note:
        Requires the credentials to have permissions to read user profile from
        Microsoft Graph (typically User.Read scope).
    """
    import requests
    from azure.identity import DefaultAzureCredential
    
    # Request a token for Microsoft Graph API
    # The .default scope requests all permissions granted to the application
    token = credentials.get_token("https://graph.microsoft.com/.default")

    # Construct HTTP headers with the Bearer token for authentication
    headers = {
        "Authorization": f"Bearer {token.token}",
        "Content-Type": "application/json"
    }

    # Call Microsoft Graph API to get current user's profile information
    response = requests.get(
        "https://graph.microsoft.com/v1.0/me",
        headers=headers
    )

    # Parse the JSON response containing user profile data
    user = response.json()
    
    # Display user information with colored formatting for better readability
    print(f"{CYAN}User display name:{RESET} {GREEN}{user.get('displayName')}{RESET}")
    print(f"{CYAN}User object ID:{RESET} {YELLOW}{user.get('id')}{RESET}")
    print(f"{CYAN}User email:{RESET} {MAGENTA}{user.get('mail') or user.get('userPrincipalName')}{RESET}")


def get_credentials(config_path) -> CertificateCredential:
    """
    Obtain and return Azure credentials using certificate-based authentication.
    
    This function loads environment variables from a .env file, then uses certificate
    authentication with a service principal to obtain Azure credentials. Certificate
    authentication is preferred for production/CI-CD scenarios as it doesn't require
    storing client secrets.
    
    Args:
        config_path: Configuration file path (currently unused, maintained for API compatibility).
                    Can be None.
    
    Returns:
        CertificateCredential: An Azure credential object that can be used to authenticate
                              against Azure services.
    
    Environment Variables Required (loaded from .azure/mlops-cdr-demo/.env):
        - AZURE_TENANT_ID: The Azure AD tenant ID where the service principal is registered
        - AZURE_SUBSCRIPTION_ID: The Azure subscription ID for resource access
        - AZURE_CLIENT_ID: The application (client) ID of the service principal
    
    Certificate Configuration:
        The function uses a hardcoded certificate path for infrastructure authentication.
        TODO: Move certificate path to environment configuration for flexibility.
    
    Raises:
        RuntimeError: If credential acquisition fails, typically due to:
                     - Invalid/missing environment variables
                     - Invalid certificate path or content
                     - Service principal misconfiguration
                     - Network connectivity issues
    
    Side Effects:
        - Loads environment variables from .env file
        - Prints configuration details and token expiry information to console
    """
    # Print current working directory for debugging purposes
    print(os.getcwd())

    # Load environment variables from the project-specific .env file
    # override=True ensures that existing environment variables are overwritten
    load_dotenv(override=True, dotenv_path="./.azure/mlops-cdr-demo/.env")
    
    # Retrieve Azure configuration from environment variables
    TENANT_ID = os.getenv("AZURE_TENANT_ID")
    print(f"{CYAN}using TENANT_ID: '{RESET} {YELLOW}{TENANT_ID}'")
    
    SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
    print(f"{CYAN}using SUBSCRIPTION_ID: '{RESET} {YELLOW}{SUBSCRIPTION_ID}'")
    
    # Client ID for the 'cdr-foundry-client' service principal
    CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
    print(f"{CYAN}using CLIENT_ID: '{RESET} {YELLOW}{CLIENT_ID}'")
    print(f"{RESET}")

    # TODO: Move certificate path configuration to .azure/{proj}/.env for flexibility
    # Certificate path for infrastructure-level authentication
    # This certificate should be associated with the service principal in Azure AD
    CERT_PATH = Path(r"C:\Carlo\Azure\certAuth\local\certs\azure\infracert_token.pem")
    print(f"using certificate from: '{CERT_PATH}'")

    try:
        # Create a CertificateCredential using the service principal configuration
        # This credential type uses certificate-based authentication instead of client secrets
        credential = CertificateCredential(
            tenant_id=TENANT_ID,
            client_id=CLIENT_ID,
            certificate_path=CERT_PATH
        )
        
        # Validate the credential by requesting a token for Azure Resource Manager
        # This ensures the certificate is valid and the service principal has proper permissions
        token = credential.get_token("https://management.azure.com/.default")
        print("Successfully obtained credentials using DefaultAzureCredential.")
        
        # Display token expiration time for monitoring purposes
        expiry = datetime.fromtimestamp(token.expires_on)
        print("Access token expires at:", expiry)
        
        return credential
        
    except Exception as ex:
        # Log the error and provide actionable guidance
        print("Failed to obtain Client Cert Credentials.", ex)
        raise RuntimeError(
            "Failed to obtain Azure credentials: Please check your configuration and environment."
        )


def getMLClient(config_path: str):
    """
    Create and return an Azure Machine Learning client for model operations.
    
    This function initializes an MLClient that can be used to interact with Azure ML
    resources including models, datasets, compute targets, and deployments.
    
    Args:
        config_path: Configuration file path (currently unused, maintained for API compatibility).
                    Can be None.
    
    Returns:
        MLClient: An authenticated Azure Machine Learning client configured for the
                 specified workspace.
    
    Environment Variables Required:
        - AZURE_SUBSCRIPTION_ID: Azure subscription containing the ML workspace
        - AZURE_ML_WORKSPACE: Name of the Azure ML workspace
        - AZURE_RESOURCE_GROUP: Resource group containing the ML workspace
    
    Usage Example:
        ml_client = getMLClient(None)
        # Use ml_client to interact with Azure ML resources
        # e.g., ml_client.models.get(name, version)
    """
    # Obtain Azure credentials using certificate authentication
    credentials = get_credentials(None)
    
    # Retrieve Azure ML workspace configuration from environment variables
    SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
    print(f"{CYAN}Using subscription:{RESET} {YELLOW}{SUBSCRIPTION_ID}{RESET}")
    
    WORKSPACE_NAME = os.environ["AZURE_ML_WORKSPACE"]
    print(f"{CYAN}Using workspace:{RESET} {GREEN}{WORKSPACE_NAME}{RESET}")
    
    RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP"]
    print(f"{CYAN}Using resource group:{RESET} {MAGENTA}{RESOURCE_GROUP}{RESET}")

    # Initialize the MLClient with credentials and workspace configuration
    # This client provides access to all Azure ML operations
    ml_client = MLClient(
        credentials,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    
    return ml_client


if __name__ == "__main__":
    # When run as a script, test the authentication flow and display results
    print(f"{CYAN}Testing Azure ML authentication...{RESET}")
    
    # Attempt to create an ML client (this will trigger the full authentication flow)
    mlclient = getMLClient(None)
    
    # Display the successfully created client object
    print(f"{CYAN}Successfully obtained credentials:{RESET} {BLUE}", mlclient)