import logging
from datetime import datetime
from bson import ObjectId

from openai import OpenAI
from temporalio import activity

from models import BatchRequest, BatchError
from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

@activity.defn
async def handle_batch_error(batch_request: BatchRequest, api_key: str) -> BatchError:
    """
    Handle errors for failed batch requests.
    
    Args:
        batch_request: Failed BatchRequest
        api_key: OpenAI API key
        
    Returns:
        BatchError with error details
        
    Raises:
        Exception: For MongoDB errors
    """
    activity.logger.info(f"Handling error for batch {batch_request.batch_id}")
    
    # Initialize client and database
    client = OpenAI(api_key=api_key)
    mongo_client = get_mongo_client()
    db = mongo_client.patent_negation
    
    # Determine error type
    error_message = batch_request.error or "Unknown error"
    error_type = "rate_limit" if "rate limit" in error_message.lower() else "processing_error"
    
    # Create batch error record
    batch_error = BatchError(
        batch_id=batch_request.batch_id,
        file_id=batch_request.file_id,
        error_type=error_type,
        error_message=error_message,
        timestamp=datetime.now(),
    )
    
    try:
        # Record error in MongoDB
        errors_collection = db.failed_batches
        result = errors_collection.insert_one(batch_error.model_dump())
        
        # Update batch status
        batch_collection = db.batch_requests
        batch_collection.update_one(
            {"_id": ObjectId(batch_request.mongodb_id)},
            {"$set": {"status": "failed", "error": error_message}}
        )
        
        # For rate limit errors, cancel batch and reset file status
        if error_type == "rate_limit":
            activity.logger.info(f"Rate limit error detected for batch {batch_request.batch_id}, cancelling")
            
            try:
                # Try to cancel the batch if possible
                client.batches.cancel(batch_request.batch_id)
            except Exception as e:
                activity.logger.error(f"Failed to cancel batch {batch_request.batch_id}: {str(e)}")
            
            # Reset file status to allow resubmission
            files_collection = db.openai_files  # Changed from "files" to "openai_files"
            files_collection.update_one(
                {"_id": ObjectId(batch_request.file_id)},
                {"$set": {"status": "uploaded", "batch_id": None}}
            )
        
        activity.logger.info(f"Successfully recorded error for batch {batch_request.batch_id}")
        return batch_error
        
    except Exception as e:
        activity.logger.error(f"Error handling batch error: {str(e)}")
        raise
