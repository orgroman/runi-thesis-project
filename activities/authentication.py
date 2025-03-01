import logging
import os
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from temporalio import activity

logger = logging.getLogger(__name__)

@activity.defn
async def authenticate_with_azure() -> Optional[str]:
    """
    Authenticate with Azure and retrieve OpenAI API key from Key Vault.
    
    Returns:
        str: OpenAI API key if successful
        
    Raises:
        Exception: If authentication or key retrieval fails
    """
    activity.logger.info("Authenticating with Azure Key Vault")
    
    try:
        # Initialize Azure credentials
        credential = DefaultAzureCredential()
        
        # Create a secret client for Key Vault
        vault_url = "https://kvrunithesis.vault.azure.net/"
        secret_client = SecretClient(vault_url=vault_url, credential=credential)
        
        # Get the OpenAI API key secret
        secret = secret_client.get_secret("alon-thesis-openai-key")
        api_key = secret.value
        
        # Set as environment variable for OpenAI client
        os.environ["OPENAI_API_KEY"] = api_key
        
        activity.logger.info("Successfully retrieved OpenAI API key from Azure Key Vault")
        return api_key
        
    except Exception as e:
        activity.logger.error(f"Error retrieving secret from Key Vault: {str(e)}")
        raise
