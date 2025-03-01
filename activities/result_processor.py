import json
import logging
import tempfile
from datetime import datetime
from bson import ObjectId
from typing import List, Dict, Any

from openai import AsyncOpenAI  # Changed from OpenAI to AsyncOpenAI
from temporalio import activity

from models import BatchRequest, ProcessingResult
from mongodb_async import get_async_database

logger = logging.getLogger(__name__)

@activity.defn
async def process_batch_results(batch_request: BatchRequest, api_key: str) -> ProcessingResult:
    """
    Process the results of a completed batch request.
    
    Args:
        batch_request: Completed BatchRequest with output file ID
        api_key: OpenAI API key
        
    Returns:
        ProcessingResult with success and error counts
        
    Raises:
        ValueError: If batch is not in completed state or missing output file
        Exception: For OpenAI API errors or MongoDB errors
    """
    activity.logger.info(f"Processing results for batch {batch_request.batch_id}")
    
    # Validate batch state
    if batch_request.status != "completed" or not batch_request.output_file_id:
        raise ValueError(f"Batch {batch_request.batch_id} is not in completed state or missing output file")
    
    # Initialize AsyncOpenAI client and MongoDB
    client = AsyncOpenAI(api_key=api_key)  # Changed to AsyncOpenAI
    db = await get_async_database()
    results_collection = db.negation_results
    batch_collection = db.batch_requests
    
    try:
        # Download results to temp file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jsonl', delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Stream file content to temp file
            response = await client.files.content(batch_request.output_file_id)  # Added await
            async for chunk in response.iter_bytes():  # Updated for async iteration
                temp_file.write(chunk)
        
        # Process JSONL results
        success_count = 0
        error_count = 0
        results = []
        
        with open(temp_path, 'r') as f:
            for line in f:
                result = json.loads(line)
                
                # Check if result has an error
                if 'error' in result:
                    error_count += 1
                    continue
                
                # Extract useful data
                try:
                    custom_id = result.get('custom_id', '')
                    # Parse response content - assuming it's already JSON
                    response_data = json.loads(result['response']['content'])
                    
                    # Create result document
                    result_doc = {
                        "custom_id": custom_id,
                        "batch_id": batch_request.batch_id,
                        "negation_present": response_data.get('negation_present', False),
                        "negation_types": response_data.get('negation_types', []),
                        "explanation": response_data.get('short_explanation', ''),
                        "processed_at": datetime.now()
                    }
                    
                    # Save to MongoDB using async
                    await results_collection.insert_one(result_doc)
                    success_count += 1
                    results.append(result_doc)
                    
                except Exception as e:
                    activity.logger.error(f"Error processing result: {str(e)}")
                    error_count += 1
        
        # Update batch status with async MongoDB
        await batch_collection.update_one(
            {"_id": ObjectId(batch_request.mongodb_id)},
            {"$set": {
                "status": "processed", 
                "processed_at": datetime.now(),
                "success_count": success_count,
                "error_count": error_count
            }}
        )
        
        # Create processing result
        processing_result = ProcessingResult(
            batch_id=batch_request.batch_id,
            output_file_id=batch_request.output_file_id,
            success_count=success_count,
            error_count=error_count,
            processed_at=datetime.now()
        )
        
        activity.logger.info(f"Successfully processed {success_count} results with {error_count} errors for batch {batch_request.batch_id}")
        return processing_result
        
    except Exception as e:
        activity.logger.error(f"Error processing batch results: {str(e)}")
        raise
