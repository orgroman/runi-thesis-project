import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
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
    'retry_delay': timedelta(minutes=2),
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

def get_batches_to_monitor(**kwargs):
    """Find in-progress batches that need monitoring."""
    client = get_mongo_client()
    db = client.patent_negation
    
    # Get batches in progress, sorted by submission time (oldest first)
    active_statuses = ["in_progress", "validating", "finalizing"]
    batches = list(db.batch_requests.find({
        "status": {"$in": active_statuses}
    }).sort("submitted_at", 1))
    
    logger.info(f"Found {len(batches)} active batches to monitor")
    
    # Return batch IDs and their MongoDB IDs
    return [(batch["batch_id"], str(batch["_id"])) for batch in batches]

def check_batch_status(batch_tuple, **kwargs):
    """Check the status of a specific batch with OpenAI API."""
    openai_batch_id, mongodb_id = batch_tuple
    client = get_mongo_client()
    db = client.patent_negation
    openai_client = get_openai_client()
    
    try:
        # Retrieve batch from OpenAI
        response = openai_client.batches.retrieve(openai_batch_id)
        current_status = response.status
        
        # Update in MongoDB
        update_data = {
            "status": current_status,
            "last_checked": datetime.now()
        }
        
        # If completed, store output file ID
        if current_status == "completed" and hasattr(response, "output_file_id"):
            update_data["output_file_id"] = response.output_file_id
            logger.info(f"Batch {openai_batch_id} completed with output file {response.output_file_id}")
        
        # If failed, store error
        if current_status == "failed" and hasattr(response, "error"):
            update_data["error"] = response.error
            logger.warning(f"Batch {openai_batch_id} failed with error: {response.error}")
        
        db.batch_requests.update_one(
            {"_id": ObjectId(mongodb_id)},
            {"$set": update_data}
        )
        
        # Return tuple of (openai_batch_id, mongodb_id, status)
        return (openai_batch_id, mongodb_id, current_status)
    
    except Exception as e:
        logger.error(f"Error checking batch {openai_batch_id}: {str(e)}")
        return (openai_batch_id, mongodb_id, "error")

def determine_next_step(batch_result_tuple, **kwargs):
    """Determine the next processing step based on batch status."""
    _, mongodb_id, status = batch_result_tuple
    
    if status == "completed":
        return "process_completed_batch"
    elif status in ["failed", "expired", "cancelled"]:
        return "handle_failed_batch"
    elif status == "error":
        return "handle_api_error"
    else:
        # Still in progress, no further action needed
        return "do_nothing"

def process_completed_batch(batch_result_tuple, **kwargs):
    """Mark a batch for result processing."""
    openai_batch_id, mongodb_id, _ = batch_result_tuple
    logger.info(f"Batch {openai_batch_id} (MongoDB ID: {mongodb_id}) marked for result processing")
    
    # No action needed here - the result_processor_dag will handle this
    return f"Batch {openai_batch_id} ready for processing"

def handle_failed_batch(batch_result_tuple, **kwargs):
    """Handle a failed batch by updating MongoDB and logging the failure."""
    openai_batch_id, mongodb_id, status = batch_result_tuple
    client = get_mongo_client()
    db = client.patent_negation
    
    try:
        # Get the batch record
        batch = db.batch_requests.find_one({"_id": ObjectId(mongodb_id)})
        
        if not batch:
            logger.error(f"Failed to find batch record {mongodb_id}")
            return f"Error: Batch record {mongodb_id} not found"
        
        # Get the associated file
        file_id = batch.get("file_id")
        if file_id:
            # Reset file status for retry
            db.openai_files.update_one(
                {"_id": ObjectId(file_id)},
                {"$set": {"status": "uploaded", "batch_id": None}}
            )
            logger.info(f"Reset file {file_id} status to 'uploaded' for retry")
        
        # Update batch record
        db.batch_requests.update_one(
            {"_id": ObjectId(mongodb_id)},
            {"$set": {"processed_at": datetime.now()}}
        )
        
        return f"Handled failed batch {openai_batch_id} (MongoDB ID: {mongodb_id})"
        
    except Exception as e:
        logger.error(f"Error handling failed batch {openai_batch_id}: {str(e)}")
        return f"Error handling failed batch {openai_batch_id}: {str(e)}"

def handle_api_error(batch_result_tuple, **kwargs):
    """Handle API errors when checking batch status."""
    openai_batch_id, mongodb_id, _ = batch_result_tuple
    logger.warning(f"API error occurred when checking batch {openai_batch_id}")
    
    # No action taken - we'll retry on next execution
    return f"Will retry batch {openai_batch_id} check on next execution"

def do_nothing(**kwargs):
    """Do nothing for in-progress batches."""
    return "No action needed for in-progress batches"

# Create the DAG
with DAG(
    'batch_monitor',
    default_args=default_args,
    description='Monitor batch request statuses',
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(1),
    catchup=False,
    tags=['openai', 'batch'],
) as dag:
    
    # Task 1: Find batches to monitor
    get_batches_task = PythonOperator(
        task_id='get_batches_to_monitor',
        python_callable=get_batches_to_monitor,
        provide_context=True,
    )
    
    # Task 2: Check each batch's status
    check_status_task = PythonOperator(
        task_id='check_batch_statuses',
        python_callable=lambda **kwargs: [
            check_batch_status(batch_tuple) 
            for batch_tuple in kwargs['ti'].xcom_pull(task_ids='get_batches_to_monitor')
        ],
        provide_context=True,
    )
    
    # Task 3: Determine next steps for each batch
    determine_next_steps_task = PythonOperator(
        task_id='determine_next_steps',
        python_callable=lambda **kwargs: [
            determine_next_step(batch_result)
            for batch_result in kwargs['ti'].xcom_pull(task_ids='check_batch_statuses')
        ],
        provide_context=True,
    )
    
    # Task 4a: Process completed batches
    process_completed_task = PythonOperator(
        task_id='process_completed_batch',
        python_callable=lambda **kwargs: [
            process_completed_batch(batch_result)
            for batch_result, next_step in zip(
                kwargs['ti'].xcom_pull(task_ids='check_batch_statuses'),
                kwargs['ti'].xcom_pull(task_ids='determine_next_steps')
            )
            if next_step == 'process_completed_batch'
        ],
        provide_context=True,
    )
    
    # Task 4b: Handle failed batches
    handle_failed_task = PythonOperator(
        task_id='handle_failed_batch',
        python_callable=lambda **kwargs: [
            handle_failed_batch(batch_result)
            for batch_result, next_step in zip(
                kwargs['ti'].xcom_pull(task_ids='check_batch_statuses'),
                kwargs['ti'].xcom_pull(task_ids='determine_next_steps')
            )
            if next_step == 'handle_failed_batch'
        ],
        provide_context=True,
    )
    
    # Task 4c: Handle API errors
    handle_api_error_task = PythonOperator(
        task_id='handle_api_error',
        python_callable=lambda **kwargs: [
            handle_api_error(batch_result)
            for batch_result, next_step in zip(
                kwargs['ti'].xcom_pull(task_ids='check_batch_statuses'),
                kwargs['ti'].xcom_pull(task_ids='determine_next_steps')
            )
            if next_step == 'handle_api_error'
        ],
        provide_context=True,
    )
    
    # Task 4d: Do nothing for in-progress batches
    do_nothing_task = PythonOperator(
        task_id='do_nothing',
        python_callable=do_nothing,
        provide_context=True,
    )
    
    # Define task dependencies
    get_batches_task >> check_status_task >> determine_next_steps_task
    determine_next_steps_task >> [process_completed_task, handle_failed_task, handle_api_error_task, do_nothing_task]
