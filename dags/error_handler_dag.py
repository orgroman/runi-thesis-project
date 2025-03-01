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
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
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

def get_failed_batches(**kwargs):
    """Find failed batches that need handling."""
    client = get_mongo_client()
    db = client.patent_negation
    
    # Get failed batches that haven't been processed
    batches = list(db.batch_requests.find({
        "status": {"$in": ["failed", "expired", "cancelled"]},
        "processed_at": {"$exists": False}
    }).sort("last_checked", 1))
    
    logger.info(f"Found {len(batches)} failed batches to handle")
    
    return [(batch["batch_id"], str(batch["_id"]), batch.get("error"), batch.get("file_id")) 
            for batch in batches]

def handle_rate_limit_error(**kwargs):
    """Find and handle rate limit errors specifically."""
    client = get_mongo_client()
    db = client.patent_negation
    
    # Find batches with rate limit errors
    rate_limit_batches = list(db.batch_requests.find({
        "$or": [
            {"error": {"$regex": "rate.*limit", "$options": "i"}},
            {"error": {"$regex": "429", "$options": "i"}}
        ],
        "processed_at": {"$exists": False}
    }))
    
    logger.info(f"Found {len(rate_limit_batches)} batches with rate limit errors")
    
    for batch in rate_limit_batches:
        try:
            # Try to cancel the batch
            if batch.get("batch_id"):
                openai_client = get_openai_client()
                try:
                    openai_client.batches.cancel(batch["batch_id"])
                    logger.info(f"Cancelled batch {batch['batch_id']} due to rate limit")
                except Exception as e:
                    logger.warning(f"Could not cancel batch {batch['batch_id']}: {str(e)}")
            
            # Reset file status for resubmission
            if batch.get("file_id"):
                db.openai_files.update_one(
                    {"_id": ObjectId(batch["file_id"])},
                    {"$set": {"status": "uploaded", "batch_id": None}}
                )
            
            # Mark batch as processed
            db.batch_requests.update_one(
                {"_id": batch["_id"]},
                {"$set": {"processed_at": datetime.now(), "handled_as": "rate_limit"}}
            )
            
        except Exception as e:
            logger.error(f"Error handling rate limit batch {batch['_id']}: {str(e)}")
    
    return f"Handled {len(rate_limit_batches)} rate-limited batches"

def handle_failed_batch(batch_tuple, **kwargs):
    """Handle a failed batch by canceling it and resetting the file status."""
    openai_batch_id, mongodb_id, error, file_id = batch_tuple
    client = get_mongo_client()
    db = client.patent_negation
    openai_client = get_openai_client()
    
    try:
        logger.info(f"Handling failed batch {openai_batch_id}")
        
        # Check if rate limit error
        is_rate_limit = False
        if error and ("rate limit" in error.lower() or "429" in error):
            is_rate_limit = True
            logger.info(f"Batch {openai_batch_id} failed due to rate limiting")
        
        # Try to cancel the batch if still possible
        try:
            openai_client.batches.cancel(openai_batch_id)
            logger.info(f"Cancelled batch {openai_batch_id}")
        except Exception as e:
            logger.warning(f"Could not cancel batch {openai_batch_id}: {str(e)}")
        
        # Reset file status for resubmission
        if file_id:
            db.openai_files.update_one(
                {"_id": ObjectId(file_id)},
                {"$set": {"status": "uploaded", "batch_id": None}}
            )
            logger.info(f"Reset file {file_id} status to 'uploaded' for resubmission")
        
        # Mark batch as processed with error details
        handled_as = "rate_limit" if is_rate_limit else "general_failure"
        db.batch_requests.update_one(
            {"_id": ObjectId(mongodb_id)},
            {"$set": {
                "processed_at": datetime.now(),
                "handled_as": handled_as
            }}
        )
        
        return f"Handled failed batch {openai_batch_id} as {handled_as}"
        
    except Exception as e:
        logger.error(f"Error handling failed batch {openai_batch_id}: {str(e)}")
        return f"Error handling failed batch {openai_batch_id}: {str(e)}"

# Create the DAG
with DAG(
    'error_handler',
    default_args=default_args,
    description='Handle failed batch requests',
    schedule_interval=timedelta(minutes=10),
    start_date=days_ago(1),
    catchup=False,
    tags=['openai', 'batch'],
) as dag:
    
    # Task 1: Handle rate limit errors specifically
    handle_rate_limit_task = PythonOperator(
        task_id='handle_rate_limit_errors',
        python_callable=handle_rate_limit_error,
        provide_context=True,
    )
    
    # Task 2: Find all failed batches
    get_failed_batches_task = PythonOperator(
        task_id='get_failed_batches',
        python_callable=get_failed_batches,
        provide_context=True,
    )
    
    # Task 3: Handle each failed batch
    handle_failed_batches_task = PythonOperator(
        task_id='handle_failed_batches',
        python_callable=lambda **kwargs: [
            handle_failed_batch(batch_tuple) 
            for batch_tuple in kwargs['ti'].xcom_pull(task_ids='get_failed_batches')
        ],
        provide_context=True,
    )
    
    # Define task dependencies
    handle_rate_limit_task >> get_failed_batches_task >> handle_failed_batches_task