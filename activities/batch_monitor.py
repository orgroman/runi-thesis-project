import logging
import time
from datetime import datetime
from bson import ObjectId

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from temporalio import activity

from models import BatchRequest
from mongodb_async import get_async_database  # Fix import to use the async database

logger = logging.getLogger(__name__)

@activity.defn
@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=300),
)
async def monitor_batch_status(batch_request: BatchRequest, api_key: str) -> BatchRequest:
    """
    Monitor a batch request until it completes, fails, or expires.
    
    Args:
        batch_request: BatchRequest to monitor
        api_key: OpenAI API key
        
    Returns:
        Updated BatchRequest with final status
        
    Raises:
        Exception: For OpenAI API errors
    """
    activity.logger.info(f"Monitoring batch {batch_request.batch_id}")
    
    # Initialize AsyncOpenAI client and get the async database
    client = AsyncOpenAI(api_key=api_key)
    db = await get_async_database()  # Use Motor async DB
    batch_collection = db.batch_requests
    
    # Initial check to ensure the batch exists
    try:
        response = await client.batches.retrieve(batch_request.batch_id)
        batch_request.status = response.status
        
        # Update in MongoDB using async
        await batch_collection.update_one(
            {"_id": ObjectId(batch_request.mongodb_id)},
            {"$set": {
                "status": batch_request.status,
                "last_checked": datetime.now()
            }}
        )
    except Exception as e:
        activity.logger.error(f"Error retrieving batch {batch_request.batch_id}: {str(e)}")
        batch_request.status = "error"
        batch_request.error = str(e)
        raise
    
    # Monitor until completion or failure
    terminal_statuses = ["completed", "failed", "expired", "cancelled"]
    check_interval = 300  # 5 minutes
    
    while batch_request.status not in terminal_statuses:
        # Signal that we're still working (for Temporal heartbeating)
        activity.heartbeat()
        
        # Wait before checking again
        time.sleep(check_interval)
        
        try:
            # Check batch status
            response = await client.batches.retrieve(batch_request.batch_id)
            batch_request.status = response.status
            batch_request.last_checked = datetime.now()
            
            # If completed, store output file ID
            if response.status == "completed" and hasattr(response, 'output_file_id'):
                batch_request.output_file_id = response.output_file_id
            
            # If failed, store error
            if response.status == "failed" and hasattr(response, 'error'):
                batch_request.error = response.error
            
            # Update in MongoDB using async
            update_fields = {
                "status": batch_request.status,
                "last_checked": batch_request.last_checked
            }
            
            if batch_request.output_file_id:
                update_fields["output_file_id"] = batch_request.output_file_id
                
            if batch_request.error:
                update_fields["error"] = batch_request.error
            
            await batch_collection.update_one(
                {"_id": ObjectId(batch_request.mongodb_id)},
                {"$set": update_fields}
            )
            
            activity.logger.info(f"Batch {batch_request.batch_id} status: {batch_request.status}")
            
        except Exception as e:
            activity.logger.error(f"Error checking batch {batch_request.batch_id}: {str(e)}")
            # Don't exit the loop, just log and try again later
    
    activity.logger.info(f"Batch {batch_request.batch_id} reached terminal status: {batch_request.status}")
    return batch_request