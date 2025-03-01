import json
import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from temporalio import activity

from models import BatchRequest, ProcessingResult, NegationResponse
from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

@activity.defn
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def process_batch_results(batch_request: BatchRequest, api_key: str) -> ProcessingResult:
    """
    Process and store results from a completed batch.
    
    Args:
        batch_request: Completed BatchRequest with output_file_id
        api_key: OpenAI API key
        
    Returns:
        ProcessingResult with statistics
        
    Raises:
        ValueError: If batch is not completed or has no output_file_id
        Exception: For OpenAI API or processing errors
    """
    activity.logger.info(f"Processing results for batch {batch_request.batch_id}")
    
    # Validate batch request
    if batch_request.status != "completed":
        raise ValueError(f"Cannot process results for non-completed batch {batch_request.batch_id}")
        
    if not batch_request.output_file_id:
        raise ValueError(f"Batch {batch_request.batch_id} has no output file ID")
    
    # Initialize clients
    client = OpenAI(api_key=api_key)
    mongo_client = get_mongo_client()
    db = mongo_client.patent_negation
    
    try:
        # Download batch results
        activity.logger.info(f"Downloading results from file {batch_request.output_file_id}")
        response = client.files.retrieve_content(batch_request.output_file_id)
        
        # Save raw results to MongoDB
        raw_results_collection = db.batch_results_raw
        raw_result_id = raw_results_collection.insert_one({
            "batch_id": batch_request.batch_id,
            "content": response,
            "processed_at": activity.info().started_at,
        }).inserted_id
        
        # Process each line (assuming response is JSONL)
        results = []
        error_count = 0
        success_count = 0
        
        for line in response.strip().split('\n'):
            try:
                result_json = json.loads(line)
                
                # Extract the custom_id to identify the record
                custom_id = result_json.get('custom_id', '')
                
                # Extract the negation analysis from the response
                if 'body' in result_json:
                    response_content = result_json['body']
                    
                    # Parse the response into our NegationResponse model
                    negation_data = NegationResponse(**response_content)
                    
                    # Store in MongoDB
                    results_collection = db.negation_results
                    result_id = results_collection.insert_one({
                        "custom_id": custom_id,
                        "batch_id": batch_request.batch_id,
                        "negation_present": negation_data.negation_present,
                        "negation_types": negation_data.negation_types,
                        "explanation": negation_data.short_explanation,
                        "processed_at": activity.info().started_at,
                    }).inserted_id
                    
                    results.append({
                        "custom_id": custom_id,
                        "result_id": str(result_id),
                        "negation_present": negation_data.negation_present
                    })
                    
                    success_count += 1
                else:
                    # Handle error case
                    error_collection = db.processing_errors
                    error_collection.insert_one({
                        "custom_id": custom_id,
                        "batch_id": batch_request.batch_id,
                        "error": "Missing response body",
                        "raw_response": result_json,
                        "processed_at": activity.info().started_at,
                    })
                    error_count += 1
                    
            except Exception as e:
                # Log and store parsing errors
                activity.logger.error(f"Error processing result line: {str(e)}")
                error_collection = db.processing_errors
                error_collection.insert_one({
                    "batch_id": batch_request.batch_id,
                    "error": str(e),
                    "raw_line": line,
                    "processed_at": activity.info().started_at,
                })
                error_count += 1
        
        # Create and store processing result
        processing_result = ProcessingResult(
            batch_id=batch_request.batch_id,
            file_id=batch_request.file_id,
            success_count=success_count,
            error_count=error_count,
            raw_result_id=str(raw_result_id),
        )
        
        # Update batch status
        batch_collection = db.batch_requests
        batch_collection.update_one(
            {"_id": batch_request.mongodb_id},
            {"$set": {
                "status": "processed",
                "processed_at": activity.info().started_at,
                "success_count": success_count,
                "error_count": error_count
            }}
        )
        
        activity.logger.info(f"Successfully processed {success_count} results with {error_count} errors")
        return processing_result
        
    except Exception as e:
        activity.logger.error(f"Error processing batch results: {str(e)}")
        raise
