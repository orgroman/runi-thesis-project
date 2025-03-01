import logging
import tempfile
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
    'retries': 5,
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

def get_files_to_upload(**kwargs):
    """Find files with 'ready' status that need to be uploaded to OpenAI."""
    client = get_mongo_client()
    db = client.patent_negation
    
    # Get files ready for upload that haven't been uploaded yet
    files = list(db.openai_files.find({
        "status": "ready",
        "openai_file_id": {"$exists": False}
    }).sort("created_at", 1))
    
    logger.info(f"Found {len(files)} files ready for upload to OpenAI")
    
    # Return file IDs and jsonl_batch_ids
    return [(str(file["_id"]), file.get("jsonl_batch_id")) for file in files]

def upload_file_to_openai(file_tuple, **kwargs):
    """Upload a file to OpenAI."""
    file_id, jsonl_batch_id = file_tuple
    client = get_mongo_client()
    db = client.patent_negation
    openai_client = get_openai_client()
    
    try:
        # Get file data from MongoDB
        file = db.openai_files.find_one({"_id": ObjectId(file_id)})
        
        if not file:
            logger.error(f"File {file_id} not found in MongoDB")
            return None
        
        # Get JSONL content from batch
        jsonl_batch = None
        if jsonl_batch_id:
            jsonl_batch = db.jsonl_batches.find_one({"_id": ObjectId(jsonl_batch_id)})
        
        if not jsonl_batch or "content" not in jsonl_batch:
            logger.error(f"JSONL batch {jsonl_batch_id} not found or has no content")
            return None
        
        # Write content to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
            temp_file.write(jsonl_batch["content"])
            temp_file_path = temp_file.name
            
        # Upload file to OpenAI
        with open(temp_file_path, "rb") as file_obj:
            response = openai_client.files.create(
                file=file_obj,
                purpose="batch"
            )
            
        # Update file in MongoDB with OpenAI file ID
        db.openai_files.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": {
                "openai_file_id": response.id,
                "status": "uploaded",
                "uploaded_at": datetime.now()
            }}
        )
        
        logger.info(f"Successfully uploaded file {file_id} to OpenAI with ID {response.id}")
        return response.id
        
    except Exception as e:
        logger.error(f"Error uploading file {file_id} to OpenAI: {str(e)}")
        
        # Update attempts count
        db.openai_files.update_one(
            {"_id": ObjectId(file_id)},
            {"$inc": {"attempts": 1}}
        )
        
        # If too many attempts, mark as failed
        file = db.openai_files.find_one({"_id": ObjectId(file_id)})
        if file and file.get("attempts", 0) >= 3:
            db.openai_files.update_one(
                {"_id": ObjectId(file_id)},
                {"$set": {"status": "upload_failed", "error": str(e)}}
            )
            logger.error(f"File {file_id} marked as failed after {file.get('attempts', 0)} attempts")
            
        return None

def process_upload_results(**kwargs):
    """Log the results of file uploads."""
    ti = kwargs['ti']
    uploaded_file_ids = ti.xcom_pull(task_ids='upload_files_to_openai')
    
    successful = sum(1 for item in uploaded_file_ids if item is not None)
    failed = sum(1 for item in uploaded_file_ids if item is None)
    
    logger.info(f"File uploads complete: {successful} successful, {failed} failed")
    return f"Uploaded {successful} files successfully"

# Create the DAG
with DAG(
    'file_upload',
    default_args=default_args,
    description='Upload files to OpenAI',
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(1),
    catchup=False,
    tags=['openai', 'file_upload'],
) as dag:
    
    # Task 1: Find files ready for upload
    get_files_task = PythonOperator(
        task_id='get_files_to_upload',
        python_callable=get_files_to_upload,
        provide_context=True,
    )
    
    # Task 2: Upload files to OpenAI
    upload_files_task = PythonOperator(
        task_id='upload_files_to_openai',
        python_callable=lambda **kwargs: [
            upload_file_to_openai(file_tuple) 
            for file_tuple in kwargs['ti'].xcom_pull(task_ids='get_files_to_upload')
        ],
        provide_context=True,
    )
    
    # Task 3: Process results
    process_results_task = PythonOperator(
        task_id='process_upload_results',
        python_callable=process_upload_results,
        provide_context=True,
    )
    
    # Define task dependencies
    get_files_task >> upload_files_task >> process_results_task
