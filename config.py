"""
Configuration settings for the OpenAI batch processing system.
"""
import os
from typing import Optional, Dict, Any

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://user:pass@localhost:27017")
DB_NAME = "patent_negation"

# MongoDB collection names
JSONL_BATCHES_COLLECTION = "jsonl_b_batches"
OPENAI_FILES_COLLECTION = "openai_files"
BATCH_REQUESTS_COLLECTION = "batch_requests"
COMPLETED_BATCHES_COLLECTION = "completed_batches"

# Azure Key Vault settings
AZURE_KEY_VAULT_URL = "https://kvrunithesis.vault.azure.net/"
OPENAI_API_KEY_SECRET_NAME = "alon-thesis-openai-key"

# OpenAI API configuration
OPENAI_BATCH_ENDPOINT = "/v1/completions"
OPENAI_COMPLETION_WINDOW = "24h"

# Polling settings
POLLING_INTERVAL_SECONDS = 300  # 5 minutes
MAX_POLLING_ATTEMPTS: Optional[int] = None  # Set to an integer for limited polls, None for unlimited
BATCH_PROCESSING_CONCURRENCY = 5  # Number of batches to process concurrently

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Temporary file storage
TEMP_FILE_DIR = "/tmp"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Get all configuration as a dictionary
def get_config() -> Dict[str, Any]:
    """Return all configuration variables as a dictionary."""
    config_dict = {k: v for k, v in globals().items() 
                  if not k.startswith('_') and k.isupper()}
    return config_dict
