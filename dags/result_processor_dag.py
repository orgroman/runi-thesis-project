import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from openai import OpenAI
from pymongo import MongoClient
from bson import ObjectId

# Configure logging
logger = logging.getLogger(__name__)

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}

def get_mongo_client():
    """Get MongoDB client with connection details from Airflow variables."""
    mongo_uri = Variable.get("mongodb_uri", "mongodb://localhost:27017/")
    return MongoClient(mongo_uri)

def get_openai_client():
    """Get OpenAI client with API key from Airflow variables."""
    openai_api_key = Variable.get("openai_api_key")
    return OpenAI(api_key=openai_api_key)

def get_completed_batches(**kwargs):
    """Find completed batches with output files ready for processing."""
    client = get_mongo_client()
    db = client.patent_negation
    
    # Get completed batches with output_file_id that haven't been processed
    batches = list(db.batch_requests.find({
        "status": "completed",
        "output_file_id": {"$exists": True, "$ne": None},
        "processed_at": {"$exists": False}
    }).sort("last_checked", 1))
    
    logger.info(f"Found {len(batches)} completed batches to process")
    
    # Return batch IDs, MongoDB IDs, and output file IDs
    return [(batch["batch_id"], str(batch["_id"]), batch["output_file_id"]) for batch in batches]

def download_and_process_results(batch_tuple, **kwargs):
    """Download and process results for a completed batch."""
    openai_batch_id, mongodb_id, output_file_id = batch_tuple
    client = get_mongo_client()
    db = client.patent_negation
    openai_client = get_openai_client()
    
    try:
        logger.info(f"Processing results for batch {openai_batch_id} with output file {output_file_id}")
        
        # Get batch information
        batch = db.batch_requests.find_one({"_id": ObjectId(mongodb_id)})
        if not batch:
            raise ValueError(f"Batch record {mongodb_id} not found in MongoDB")
        
        # Download file content
        response = openai_client.files.content(output_file_id)
        content = response.text
        
        # Parse JSONL responses
        results = []
        for line in content.strip().split('\n'):
            result = json.loads(line)
            
            # Extract custom_id from the request to match with original data
            custom_id = None
            if "request" in result and "body" in result["request"]:
                request_body = result["request"]["body"]
                if isinstance(request_body, str):
                    try:
                        request_body = json.loads(request_body)
                    except json.JSONDecodeError:
                        pass
                
                if isinstance(request_body, dict) and "custom_id" in request_body:
                    custom_id = request_body["custom_id"]
            
            # Extract response data
            response_data = None
            if "response" in result and "choices" in result["response"]:
                choices = result["response"]["choices"]
                if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                    try:
                        response_data = json.loads(choices[0]["message"]["content"])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse response as JSON: {choices[0]['message']['content']}")
            
            # Create result document
            if custom_id and response_data:
                result_doc = {
                    "custom_id": custom_id,
                    "batch_id": openai_batch_id,
                    "file_id": batch.get("file_id"),
                    "jsonl_batch_id": batch.get("jsonl_batch_id"),
                    "negation_present": response_data.get("negation_present"),
                    "negation_types": response_data.get("negation_types", []),
                    "explanation": response_data.get("short_explanation"),
                    "processed_at": datetime.now(),
                    "raw_response": result
                }
                results.append(result_doc)
        
        # Save results to MongoDB
        if results:
            db.results.insert_many(results)
            logger.info(f"Saved {len(results)} results to MongoDB")
        
        # Mark batch as processed
        db.batch_requests.update_one(
            {"_id": ObjectId(mongodb_id)},
            {"$set": {"processed_at": datetime.now(), "result_count": len(results)}}
        )
        
        # Mark file as completed
        if batch.get("file_id"):
            db.openai_files.update_one(
                {"_id": ObjectId(batch["file_id"])},
                {"$set": {"status": "completed", "completed_at": datetime.now()}}
            )
        
        return f"Processed {len(results)} results for batch {openai_batch_id}"
        
    except Exception as e:
        logger.error(f"Error processing results for batch {openai_batch_id}: {str(e)}")
        
        # Mark as error for retry
        db.batch_requests.update_one(
            {"_id": ObjectId(mongodb_id)},
            {"$set": {"processing_error": str(e), "error_at": datetime.now()}}
        )
        
        raise

# Create the DAG
with DAG(
    'result_processor',
    default_args=default_args,
    description='Process completed batch results',
    schedule_interval=timedelta(minutes=10),
    start_date=days_ago(1),
    catchup=False,
    tags=['openai', 'batch'],
) as dag:
    
    # Task 1: Find completed batches to process
    get_completed_batches_task = PythonOperator(
        task_id='get_completed_batches',
        python_callable=get_completed_batches,
        provide_context=True,
    )
    
    # Task 2: Process each batch's results
    process_results_task = PythonOperator(
        task_id='process_batch_results',
        python_callable=lambda **kwargs: [
            download_and_process_results(batch_tuple) 
            for batch_tuple in kwargs['ti'].xcom_pull(task_ids='get_completed_batches')
        ],
        provide_context=True,
    )
    
    # Define task dependencies
    get_completed_batches_task >> process_results_task
