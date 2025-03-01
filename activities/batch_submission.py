import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from temporalio import activity

from models import FileMetadata, BatchRequest
from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

@activity.defn
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def submit_batch_request(file_metadata: FileMetadata, api_key: str) -> BatchRequest:
    """
    Submit a batch processing request to OpenAI.
    
    Args:
        file_metadata: FileMetadata with OpenAI file ID
        api_key: OpenAI API key
        
    Returns:
        BatchRequest object with OpenAI batch ID
        
    Raises:
        ValueError: If file is not uploaded to OpenAI
        Exception: For OpenAI API errors
    """
    activity.logger.info(f"Submitting batch request for file {file_metadata.file_name}")
    
    # Validate file metadata
    if not file_metadata.openai_file_id:
        raise ValueError(f"File {file_metadata.file_name} is not uploaded to OpenAI")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Submit batch request
        response = client.batches.create(
            input_file_id=file_metadata.openai_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        # Create batch request object
        now = datetime.now()
        expires_at = now + timedelta(hours=24)
        
        batch_request = BatchRequest(
            file_id=file_metadata.mongodb_id,
            openai_file_id=file_metadata.openai_file_id,
            jsonl_batch_id=file_metadata.jsonl_batch_id,  # Link to JSONL batch
            batch_id=response.id,
            status="in_progress",
            created_at=now,
            expires_at=expires_at,
            last_checked=now,
        )
        
        # Save in MongoDB
        mongo_client = get_mongo_client()
        db = mongo_client.patent_negation
        batch_collection = db.batch_requests
        
        result = batch_collection.insert_one(batch_request.model_dump(exclude={"mongodb_id"}))
        batch_request.mongodb_id = str(result.inserted_id)
        
        # Update file status
        files_collection = db.files
        files_collection.update_one(
            {"_id": file_metadata.mongodb_id},
            {"$set": {"status": "processing", "batch_id": batch_request.batch_id}}
        )
        
        # Update JSONL batch status
        jsonl_collection = db.jsonl_batches
        jsonl_collection.update_one(
            {"_id": file_metadata.jsonl_batch_id},
            {"$set": {"status": "processing", "batch_id": batch_request.batch_id}}
        )
        
        activity.logger.info(f"Successfully submitted batch with ID: {response.id}")
        return batch_request
        
    except Exception as e:
        activity.logger.error(f"Error submitting batch request: {str(e)}")
        raise

@activity.defn
async def wait_for_batch_completion(
    batch_requests: List[BatchRequest], 
    api_key: str,
    check_interval: int = 300  # 5 minutes in seconds
) -> Optional[BatchRequest]:
    """
    Wait for at least one batch in the list to complete or fail.
    
    Args:
        batch_requests: List of batch requests to monitor
        api_key: OpenAI API key
        check_interval: Seconds between status checks
        
    Returns:
        The first completed or failed batch request, or None if timeout
    """
    activity.logger.info(f"Waiting for completion of {len(batch_requests)} batches")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Check batches until one completes
        while activity.info().heartbeat_details is None:  # Check if cancelled
            for batch in batch_requests:
                activity.heartbeat()  # Signal that we're still working
                
                try:
                    response = client.batches.retrieve(batch.batch_id)
                    status = response.status
                    
                    # Update batch status in MongoDB
                    mongo_client = get_mongo_client()
                    db = mongo_client.patent_negation
                    batch_collection = db.batch_requests
                    
                    batch_collection.update_one(
                        {"_id": batch.mongodb_id},
                        {"$set": {
                            "status": status,
                            "last_checked": datetime.now()
                        }}
                    )
                    
                    # If batch is completed or failed, return it
                    if status in ["completed", "failed", "expired", "cancelled"]:
                        activity.logger.info(f"Batch {batch.batch_id} is {status}")
                        return batch
                        
                except Exception as e:
                    activity.logger.error(f"Error checking batch {batch.batch_id}: {str(e)}")
            
            # Wait before checking again
            activity.logger.info(f"No completed batches, waiting {check_interval} seconds")
            time.sleep(check_interval)
        
        return None
        
    except Exception as e:
        activity.logger.error(f"Error waiting for batch completion: {str(e)}")
        raise
