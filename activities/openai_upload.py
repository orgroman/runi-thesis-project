import logging
import tempfile
import asyncio
from datetime import datetime
from pathlib import Path
from bson import ObjectId
from typing import List, Optional

from openai import AsyncOpenAI  # Changed from OpenAI to AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from temporalio import activity

from models import FileMetadata
from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

async def upload_single_file(file_metadata: FileMetadata, api_key: str) -> Optional[FileMetadata]:
    """
    Upload a single file to OpenAI.
    This function is designed to be called concurrently with asyncio.gather.
    """
    try:
        # Get MongoDB client and retrieve the JSONL content
        mongo_client = get_mongo_client()
        db = mongo_client.patent_negation
        jsonl_collection = db.jsonl_batches
        
        # Convert string ID to ObjectId for MongoDB query
        jsonl_batch = jsonl_collection.find_one({"_id": ObjectId(file_metadata.jsonl_batch_id)})
        if not jsonl_batch:
            activity.logger.error(f"JSONL batch not found with ID: {file_metadata.jsonl_batch_id}")
            return None
        
        # Create a temporary file to hold the JSONL content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
            temp_file.write(jsonl_batch["content"])
            temp_file_path = temp_file.name
        
        # Initialize AsyncOpenAI client
        client = AsyncOpenAI(api_key=api_key)  # Changed to AsyncOpenAI
        
        # Upload file to OpenAI
        with open(temp_file_path, 'rb') as file:
            response = await client.files.create(  # Added await
                file=file,
                purpose='batch'
            )
        
        # Update file metadata with OpenAI file ID
        file_metadata.openai_file_id = response.id
        file_metadata.status = "uploaded"
        file_metadata.uploaded_at = datetime.now()
        
        # Update in MongoDB
        files_collection = db.openai_files  # Changed from "files" to "openai_files"
        files_collection.update_one(
            {"_id": ObjectId(file_metadata.mongodb_id)},
            {"$set": {
                "openai_file_id": file_metadata.openai_file_id,
                "status": "uploaded",
                "uploaded_at": file_metadata.uploaded_at
            }}
        )
        
        activity.logger.info(f"Successfully uploaded file {file_metadata.file_name} to OpenAI with ID: {response.id}")
        return file_metadata
        
    except Exception as e:
        # Update failure status in MongoDB
        activity.logger.error(f"Error uploading file {file_metadata.file_name}: {str(e)}")
        
        if hasattr(file_metadata, 'mongodb_id'):
            try:
                mongo_client = get_mongo_client()
                db = mongo_client.patent_negation
                files_collection = db.openai_files  # Changed from "files" to "openai_files"
                
                files_collection.update_one(
                    {"_id": ObjectId(file_metadata.mongodb_id)},
                    {"$set": {
                        "status": "upload_failed",
                        "error": str(e),
                        "attempts": file_metadata.attempts + 1
                    }}
                )
            except Exception as mongo_err:
                activity.logger.error(f"Failed to update MongoDB for {file_metadata.file_name}: {str(mongo_err)}")
        
        return None

@activity.defn
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def upload_files_to_openai(file_metadatas: List[FileMetadata], api_key: str, 
                                max_concurrent: int = 5) -> List[FileMetadata]:
    """
    Upload multiple JSONL batches from MongoDB to OpenAI concurrently.
    
    Args:
        file_metadatas: List of FileMetadata objects to upload
        api_key: OpenAI API key
        max_concurrent: Maximum number of concurrent uploads
        
    Returns:
        List of successfully uploaded FileMetadata objects
        
    Raises:
        Exception: If all uploads fail
    """
    activity.logger.info(f"Uploading {len(file_metadatas)} files to OpenAI")
    
    # Process files in batches to control concurrency
    results = []
    for i in range(0, len(file_metadatas), max_concurrent):
        batch = file_metadatas[i:i+max_concurrent]
        activity.logger.info(f"Processing batch {i//max_concurrent + 1} of {len(file_metadatas)//max_concurrent + 1} ({len(batch)} files)")
        
        # Upload files concurrently
        upload_tasks = [upload_single_file(file_metadata, api_key) for file_metadata in batch]
        batch_results = await asyncio.gather(*upload_tasks)
        
        # Filter out failed uploads (None values)
        successful_uploads = [result for result in batch_results if result is not None]
        results.extend(successful_uploads)
        
        activity.logger.info(f"Batch {i//max_concurrent + 1} completed: {len(successful_uploads)}/{len(batch)} successful")
    
    if not results:
        raise Exception("All file uploads failed")
    
    activity.logger.info(f"Completed uploads: {len(results)}/{len(file_metadatas)} successful")
    return results

# Keep the original single file upload for backward compatibility
@activity.defn
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def upload_file_to_openai(file_metadata: FileMetadata, api_key: str) -> FileMetadata:
    """
    Upload a single JSONL batch from MongoDB to OpenAI.
    
    Args:
        file_metadata: FileMetadata object with MongoDB references
        api_key: OpenAI API key
        
    Returns:
        Updated FileMetadata with OpenAI file ID
        
    Raises:
        KeyError: If the JSONL batch does not exist
        Exception: For OpenAI API errors
    """
    activity.logger.info(f"Uploading file {file_metadata.file_name} to OpenAI")
    
    result = await upload_single_file(file_metadata, api_key)
    if result is None:
        raise Exception(f"Failed to upload file {file_metadata.file_name}")
    
    return result
