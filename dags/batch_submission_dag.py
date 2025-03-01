import logging
from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.decorators import task
from airflow.providers.python.operators.python_virtualenv import PythonVirtualenvOperator
from airflow.utils.dates import days_ago

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

# Define functions to be executed in virtualenv
def get_files_to_submit(**kwargs):
    """Find files with 'uploaded' status that need batch submission."""
    from pymongo import MongoClient
    import logging
    import os
    from bson import ObjectId
    
    logger = logging.getLogger(__name__)
    
    # Get MongoDB connection details from environment variables or use defaults
    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client.patent_negation
    
    # Get current active batch count
    active_batches = db.batch_requests.count_documents({
        "status": {"$in": ["in_progress", "validating", "finalizing"]}
    })
    
    # Check if we're below rate limits (max 50 concurrent batches)
    remaining_capacity = 45 - active_batches  # Keep a buffer of 5
    
    if remaining_capacity <= 0:
        logger.info("Maximum concurrent batch limit reached, skipping submission")
        return []
    
    # Get files ready for batch submission
    files = list(db.openai_files.find({
        "status": "uploaded",
        "openai_file_id": {"$exists": True, "$ne": None}
    }).limit(remaining_capacity))
    
    logger.info(f"Found {len(files)} files ready for batch submission")
    
    # Pass file IDs for next task
    return [str(file["_id"]) for file in files]

def submit_batch_request(file_id, **kwargs):
    """Submit a batch request to OpenAI for a single file."""
    from pymongo import MongoClient
    import logging
    import os
    from datetime import datetime
    from bson import ObjectId
    from openai import OpenAI
    
    logger = logging.getLogger(__name__)
    
    # Get MongoDB connection details from environment variables or use defaults
    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client.patent_negation
    
    # Get OpenAI API key from environment or use default (replace with your key management approach)
    openai_api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    openai_client = OpenAI(api_key=openai_api_key)
    
    try:
        # Get file data from MongoDB
        file = db.openai_files.find_one({"_id": ObjectId(file_id)})
        
        if not file or file["status"] != "uploaded" or not file.get("openai_file_id"):
            logger.warning(f"File {file_id} is not valid for submission")
            return None
        
        # Submit batch request to OpenAI
        response = openai_client.batches.create(
            file_id=file["openai_file_id"],
            purpose="negation-detection"
        )
        
        # Create batch request record
        batch_request = {
            "batch_id": response.id,
            "file_id": file_id,
            "jsonl_batch_id": file.get("jsonl_batch_id"),
            "status": response.status,
            "submitted_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24),
            "last_checked": datetime.now()
        }
        
        # Save to MongoDB
        result = db.batch_requests.insert_one(batch_request)
        
        # Update file status
        db.openai_files.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": {"status": "batch_submitted", "batch_id": response.id}}
        )
        
        logger.info(f"Successfully submitted batch request {response.id} for file {file_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"Error submitting batch request for file {file_id}: {str(e)}")
        return None

def process_submission_results(**kwargs):
    """Log the results of batch submissions."""
    ti = kwargs['ti']
    submitted_batch_ids = ti.xcom_pull(task_ids='submit_batch_requests')
    
    successful = sum(1 for item in submitted_batch_ids if item is not None)
    failed = sum(1 for item in submitted_batch_ids if item is None)
    
    logger.info(f"Batch submission complete: {successful} successful, {failed} failed")
    return f"Submitted {successful} batches successfully"

# Path to requirements file - THIS IS KEY FOR VIRTUALENV
requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

# Create the DAG
with DAG(
    'batch_submission',
    default_args=default_args,
    description='Submit batch requests to OpenAI',
    schedule_interval=timedelta(minutes=15),
    start_date=days_ago(1),
    catchup=False,
    tags=['openai', 'batch'],
) as dag:
    
    # Task 1: Find files ready for batch submission - using VirtualenvOperator
    get_files_task = PythonVirtualenvOperator(
        task_id='get_files_to_submit',
        python_callable=get_files_to_submit,
        requirements=["pymongo==4.11.1", "bson==0.5.10"],  # Specify required packages
        system_site_packages=False,
    )
    
    # Task 2: Submit batch requests for each file - using VirtualenvOperator
    submit_batch_task = PythonVirtualenvOperator(
        task_id='submit_batch_requests',
        python_callable=lambda **kwargs: [
            submit_batch_request(file_id) 
            for file_id in kwargs['ti'].xcom_pull(task_ids='get_files_to_submit')
        ],
        requirements=["pymongo==4.11.1", "bson==0.5.10", "openai==1.63.2"],
        system_site_packages=False,
    )
    
    # Task 3: Process results - this can use a regular PythonOperator as it doesn't need special packages
    process_results_task = PythonVirtualenvOperator(
        task_id='process_submission_results',
        python_callable=process_submission_results,
        requirements=[],  # No special requirements
        system_site_packages=True,
    )
    
    # Define task dependencies
    get_files_task >> submit_batch_task >> process_results_task
