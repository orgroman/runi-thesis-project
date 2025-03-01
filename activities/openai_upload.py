import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from temporalio import activity

from models import FileMetadata
from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

@activity.defn
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def upload_file_to_openai(file_metadata: FileMetadata, api_key: str) -> FileMetadata:
    """
    Upload a JSONL file to OpenAI for batch processing.
    
    Args:
        file_metadata: FileMetadata object with file info
        api_key: OpenAI API key
        
    Returns:
        Updated FileMetadata with OpenAI file ID
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For OpenAI API errors
    """
    activity.logger.info(f"Uploading file {file_metadata.file_name} to OpenAI")
    
    # Check if file exists
    file_path = Path(file_metadata.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_metadata.file_path}")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Upload file to OpenAI
        with open(file_path, 'rb') as file:
            response = client.files.create(
                file=file,
                purpose='batch'
            )
        
        # Update file metadata with OpenAI file ID
        file_metadata.openai_file_id = response.id
        file_metadata.status = "uploaded"
        file_metadata.uploaded_at = datetime.now()
        
        # Update in MongoDB
        mongo_client = get_mongo_client()
        db = mongo_client.patent_negation
        files_collection = db.files
        
        files_collection.update_one(
            {"_id": file_metadata.mongodb_id},
            {"$set": {
                "openai_file_id": file_metadata.openai_file_id,
                "status": "uploaded",
                "uploaded_at": file_metadata.uploaded_at
            }}
        )
        
        activity.logger.info(f"Successfully uploaded file to OpenAI with ID: {response.id}")
        return file_metadata
        
    except Exception as e:
        # Update failure status in MongoDB
        if hasattr(file_metadata, 'mongodb_id'):
            try:
                mongo_client = get_mongo_client()
                db = mongo_client.patent_negation
                files_collection = db.files
                
                files_collection.update_one(
                    {"_id": file_metadata.mongodb_id},
                    {"$set": {
                        "status": "upload_failed",
                        "error": str(e),
                        "attempts": file_metadata.attempts + 1
                    }}
                )
            except Exception as mongo_err:
                activity.logger.error(f"Failed to update MongoDB: {str(mongo_err)}")
        
        activity.logger.error(f"Error uploading file to OpenAI: {str(e)}")
        raise
