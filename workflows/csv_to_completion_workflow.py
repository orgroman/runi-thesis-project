"""
End-to-end workflow for processing CSV data through OpenAI batch completions.
Steps:
1. Process CSV into JSONL batches and store in MongoDB
2. Register each JSONL batch as a file for OpenAI processing
3. Upload each file to OpenAI and create batch requests
4. Monitor batch requests until completion
5. Store results in MongoDB
"""
import asyncio
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI

import config
from csv_to_jsonl_processor import process_csv_to_jsonl_batches
from retr_batch_openai import get_openai_key, handle_file_management
from activities.mongodb_registration import register_file_in_mongodb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def register_and_process_jsonl_batches(
    csv_path: str,
    output_dir: Optional[str] = None,
    batch_size: int = 1000,
    text_column: str = "text",
    poll_interval_seconds: int = 300,
    max_polling_attempts: Optional[int] = None
) -> Dict[str, Any]:
    """
    End-to-end workflow for processing CSV data through OpenAI batch completions.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to store JSONL files
        batch_size: Number of rows per batch
        text_column: Column containing text to analyze
        poll_interval_seconds: Time between polling attempts in seconds
        max_polling_attempts: Maximum number of polling attempts (None for unlimited)
        
    Returns:
        Dictionary with workflow results
    """
    start_time = datetime.now()
    workflow_results = {
        "workflow_start": start_time.isoformat(),
        "csv_path": csv_path,
        "batch_size": batch_size,
        "text_column": text_column,
        "output_dir": output_dir or str(Path(csv_path).parent / "jsonl_batches"),
        "steps": []
    }
    
    # Step 1: Process CSV into JSONL batches and store in MongoDB
    logger.info("Step 1: Processing CSV into JSONL batches")
    step_start = datetime.now()
    
    try:
        batch_metadata = await process_csv_to_jsonl_batches(
            csv_path=csv_path,
            output_dir=output_dir,
            batch_size=batch_size,
            text_column=text_column,
            mongodb_uri=config.MONGODB_URI,
            db_name=config.DB_NAME,
            collection_name=config.JSONL_BATCHES_COLLECTION
        )
        
        workflow_results["steps"].append({
            "name": "csv_to_jsonl",
            "status": "success",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "batch_count": len(batch_metadata),
            "batch_metadata": batch_metadata
        })
        
        logger.info(f"Created {len(batch_metadata)} JSONL batches")
    except Exception as e:
        logger.error(f"Error processing CSV into JSONL batches: {str(e)}")
        workflow_results["steps"].append({
            "name": "csv_to_jsonl",
            "status": "failed",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error": str(e)
        })
        workflow_results["workflow_end"] = datetime.now().isoformat()
        workflow_results["status"] = "failed"
        return workflow_results
    
    # Step 2: Register files in MongoDB
    logger.info("Step 2: Registering JSONL batches in MongoDB")
    step_start = datetime.now()
    
    try:
        mongodb_client = AsyncIOMotorClient(config.MONGODB_URI)
        files_collection = mongodb_client[config.DB_NAME][config.OPENAI_FILES_COLLECTION]
        
        registered_files = []
        for batch in batch_metadata:
            if batch.get("mongodb_id"):
                file_metadata = await register_file_in_mongodb(batch["mongodb_id"])
                registered_files.append({
                    "jsonl_batch_id": batch["mongodb_id"],
                    "file_metadata_id": file_metadata.mongodb_id
                })
        
        workflow_results["steps"].append({
            "name": "register_files",
            "status": "success",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "registered_files": registered_files
        })
        
        logger.info(f"Registered {len(registered_files)} files in MongoDB")
    except Exception as e:
        logger.error(f"Error registering files in MongoDB: {str(e)}")
        workflow_results["steps"].append({
            "name": "register_files",
            "status": "failed",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error": str(e)
        })
    
    # Step 3: Start polling for batch management
    logger.info("Step 3: Starting OpenAI file and batch management")
    step_start = datetime.now()
    
    try:
        # Get OpenAI API key
        get_openai_key()
        
        # Initialize MongoDB and OpenAI clients
        motor_client = AsyncIOMotorClient(config.MONGODB_URI)
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Start polling
        polling_results = {
            "polling_start": step_start.isoformat(),
            "poll_interval_seconds": poll_interval_seconds,
            "max_polling_attempts": max_polling_attempts,
            "attempts": []
        }
        
        attempts = 0
        while True:
            if max_polling_attempts and attempts >= max_polling_attempts:
                logger.info(f"Reached maximum polling attempts ({max_polling_attempts})")
                break
            
            logger.info(f"Polling cycle {attempts + 1}")
            attempt_start = datetime.now()
            
            try:
                await handle_file_management(openai_client, motor_client, config.DB_NAME)
                logger.info("Completed file and batch management cycle")
                
                # Check if all batches are completed
                batch_requests = motor_client[config.DB_NAME][config.BATCH_REQUESTS_COLLECTION]
                completed_batches = motor_client[config.DB_NAME][config.COMPLETED_BATCHES_COLLECTION]
                
                pending_count = await batch_requests.count_documents({})
                completed_count = await completed_batches.count_documents({})
                
                polling_results["attempts"].append({
                    "attempt": attempts + 1,
                    "start_time": attempt_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "pending_count": pending_count,
                    "completed_count": completed_count
                })
                
                if pending_count == 0 and completed_count >= len(batch_metadata):
                    logger.info(f"All batches completed! Found {completed_count} completed batches.")
                    break
                
            except Exception as e:
                logger.error(f"Error in polling cycle: {str(e)}")
                polling_results["attempts"].append({
                    "attempt": attempts + 1,
                    "start_time": attempt_start.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "error": str(e)
                })
            
            attempts += 1
            
            # Sleep before next polling attempt
            logger.info(f"Sleeping for {poll_interval_seconds} seconds before next polling attempt")
            await asyncio.sleep(poll_interval_seconds)
        
        polling_results["polling_end"] = datetime.now().isoformat()
        workflow_results["steps"].append({
            "name": "batch_management",
            "status": "success",
            "polling_details": polling_results
        })
        
    except Exception as e:
        logger.error(f"Error in batch management: {str(e)}")
        workflow_results["steps"].append({
            "name": "batch_management",
            "status": "failed",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error": str(e)
        })
    
    # Step 4: Gather final results
    logger.info("Step 4: Gathering final results")
    step_start = datetime.now()
    
    try:
        mongodb_client = AsyncIOMotorClient(config.MONGODB_URI)
        completed_batches = mongodb_client[config.DB_NAME][config.COMPLETED_BATCHES_COLLECTION]
        
        # Count completed batches and total results
        completed_count = await completed_batches.count_documents({})
        
        # Get sample of results
        sample_batch = await completed_batches.find_one({})
        sample_results = None
        if sample_batch and "results" in sample_batch and len(sample_batch["results"]) > 0:
            sample_results = sample_batch["results"][0]
        
        workflow_results["steps"].append({
            "name": "gather_results",
            "status": "success",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "completed_batches": completed_count,
            "sample_result": sample_results
        })
        
        logger.info(f"Found {completed_count} completed batches")
    except Exception as e:
        logger.error(f"Error gathering final results: {str(e)}")
        workflow_results["steps"].append({
            "name": "gather_results",
            "status": "failed",
            "start_time": step_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "error": str(e)
        })
    
    # Workflow completion
    workflow_results["workflow_end"] = datetime.now().isoformat()
    workflow_results["status"] = "completed"
    
    # Calculate duration
    workflow_duration = datetime.now() - start_time
    workflow_results["duration_seconds"] = workflow_duration.total_seconds()
    
    logger.info(f"Workflow completed in {workflow_duration}")
    
    return workflow_results

async def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-end workflow for CSV to OpenAI batch completions")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("--output-dir", help="Directory to store JSONL files")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of rows per batch")
    parser.add_argument("--text-column", default="text", help="Column containing text to analyze")
    parser.add_argument("--poll-interval", type=int, default=300, help="Time between polling attempts in seconds")
    parser.add_argument("--max-attempts", type=int, help="Maximum number of polling attempts")
    
    args = parser.parse_args()
    
    results = await register_and_process_jsonl_batches(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        text_column=args.text_column,
        poll_interval_seconds=args.poll_interval,
        max_polling_attempts=args.max_attempts
    )
    
    # Write results to file
    results_path = Path(args.output_dir or Path(args.csv_path).parent / "jsonl_batches") / "workflow_results.json"
    import json
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Workflow results written to {results_path}")

if __name__ == "__main__":
    asyncio.run(main())
