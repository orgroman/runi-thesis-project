import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional
from bson import ObjectId

from openai import AsyncOpenAI  # Changed from OpenAI to AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from temporalio import activity

from models import FileMetadata, BatchRequest
from mongodb_async import get_async_database  # Proper import for async database

logger = logging.getLogger(__name__)

@activity.defn
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
)
async def submit_batch_request(file_metadata: FileMetadata, api_key: str) -> BatchRequest:
    """
    Submit a batch request to OpenAI.
    
    Args:
        file_metadata: FileMetadata with OpenAI file ID
        api_key: OpenAI API key
        
    Returns:
        BatchRequest with batch ID and status
        
    Raises:
        ValueError: If file is not in uploaded state
        Exception: For OpenAI API errors
    """
    activity.logger.info(f"Submitting batch request for file {file_metadata.file_name}")
    
    # Validate file is in uploaded state
    if file_metadata.status != "uploaded" or not file_metadata.openai_file_id:
        raise ValueError(f"File {file_metadata.file_name} is not in uploaded state or missing OpenAI file ID")
    
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(api_key=api_key)  # Changed to AsyncOpenAI
    db = await get_async_database()
    files_collection = db.openai_files
    batch_collection = db.batch_requests
    
    try:
        # Submit batch request
        response = await client.batches.create(  # Added await
            file_id=file_metadata.openai_file_id,
            purpose="negation-detection",
            # Add any additional batch processing parameters here
        )
        
        # Create batch request record
        batch_request = BatchRequest(
            batch_id=response.id,
            file_id=file_metadata.mongodb_id,
            jsonl_batch_id=file_metadata.jsonl_batch_id,
            status=response.status,
            submitted_at=datetime.now()
        )
        
        # Save to MongoDB with async
        result = await batch_collection.insert_one(batch_request.model_dump(exclude={"mongodb_id"}))
        batch_request.mongodb_id = str(result.inserted_id)
        
        # Update file status with async
        await files_collection.update_one(
            {"_id": ObjectId(file_metadata.mongodb_id)},
            {"$set": {"status": "batch_submitted", "batch_id": batch_request.batch_id}}
        )
        
        activity.logger.info(f"Successfully submitted batch request {batch_request.batch_id} for file {file_metadata.file_name}")
        return batch_request
        
    except Exception as e:
        activity.logger.error(f"Error submitting batch request: {str(e)}")
        raise

@activity.defn
@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=300),
)
async def wait_for_batch_completion(batch_request: BatchRequest, api_key: str, timeout_seconds: int = 3600) -> BatchRequest:
    """
    Wait for a batch request to complete.
    
    Args:
        batch_request: BatchRequest to wait for
        api_key: OpenAI API key
        timeout_seconds: Maximum time to wait in seconds
        
    Returns:
        Updated BatchRequest with final status
        
    Raises:
        TimeoutError: If the batch does not complete within the timeout
        Exception: For OpenAI API errors
    """
    activity.logger.info(f"Waiting for batch {batch_request.batch_id} completion")
    
    # Initialize AsyncOpenAI client and get async database
    client = AsyncOpenAI(api_key=api_key)
    db = await get_async_database()  # Use async database access
    batch_collection = db.batch_requests
    
    start_time = time.time()
    check_interval = 60  # seconds
    
    while True:
        # Signal that we're still working (for Temporal heartbeating)
        activity.heartbeat()
        
        # Check if timeout has expired
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Batch {batch_request.batch_id} did not complete within timeout")
        
        try:
            # Check batch status
            response = await client.batches.retrieve(batch_request.batch_id)  # Added await
            batch_request.status = response.status
            
            # Update in MongoDB using async
            await batch_collection.update_one(
                {"_id": ObjectId(batch_request.mongodb_id)},
                {"$set": {"status": batch_request.status, "last_checked": datetime.now()}}
            )
            
            activity.logger.info(f"Batch {batch_request.batch_id} status: {batch_request.status}")
            
            # If completed or failed, break the loop
            if batch_request.status in ["completed", "failed", "expired", "cancelled"]:
                if batch_request.status == "completed" and hasattr(response, "output_file_id"):
                    batch_request.output_file_id = response.output_file_id
                    
                    # Update in MongoDB using async
                    await batch_collection.update_one(
                        {"_id": ObjectId(batch_request.mongodb_id)},
                        {"$set": {"output_file_id": batch_request.output_file_id}}
                    )
                    
                break
            
            # Wait before checking again
            time.sleep(check_interval)
            
        except Exception as e:
            activity.logger.error(f"Error checking batch status: {str(e)}")
            time.sleep(check_interval * 2)  # Wait longer after error
    
    activity.logger.info(f"Batch {batch_request.batch_id} completed with status: {batch_request.status}")
    return batch_request
