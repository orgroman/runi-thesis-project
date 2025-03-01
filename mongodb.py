import os
import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pymongo.database import Database

logger = logging.getLogger(__name__)

# MongoDB connection parameters
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://user:pass@localhost:27017/")
DB_NAME = os.environ.get("MONGO_DB", "patent_negation")

# Global client instance
_mongo_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    """
    Get a MongoDB client instance (singleton pattern).
    
    Returns:
        MongoDB client instance
    """
    global _mongo_client
    
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB at {MONGO_URI}")
        except ConnectionFailure as e:  # Added 'as e' to capture the exception
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
            
    return _mongo_client


def get_database() -> Database:
    """
    Get the MongoDB database instance.
    
    Returns:
        MongoDB database instance
    """
    client = get_mongo_client()
    return client[DB_NAME]


def setup_collections():
    """
    Set up MongoDB collections with indexes for better query performance.
    """
    db = get_database()
    
    # Files collection
    if "files" not in db.list_collection_names():
        db.create_collection("files")
        db.files.create_index("openai_file_id")
        db.files.create_index("status")
        db.files.create_index("jsonl_batch_id")
    
    # Batch requests collection
    if "batch_requests" not in db.list_collection_names():
        db.create_collection("batch_requests")
        db.batch_requests.create_index("batch_id")
        db.batch_requests.create_index("file_id")
        db.batch_requests.create_index("status")
        db.batch_requests.create_index("jsonl_batch_id")
    
    # Results collection
    if "negation_results" not in db.list_collection_names():
        db.create_collection("negation_results")
        db.negation_results.create_index("custom_id")
        db.negation_results.create_index("batch_id")
        db.negation_results.create_index("negation_present")
    
    # Raw results collection
    if "jsonl_batches" not in db.list_collection_names():
        db.create_collection("jsonl_batches")
        db.jsonl_batches.create_index("batch_number")
        db.jsonl_batches.create_index("status")
        db.jsonl_batches.create_index("source_dataframe")
    
    # Failed batches collection
    if "failed_batches" not in db.list_collection_names():
        db.create_collection("failed_batches")
        db.failed_batches.create_index("batch_id")
    
    logger.info("MongoDB collections and indexes set up successfully")
