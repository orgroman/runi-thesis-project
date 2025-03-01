import os
import logging
from typing import Optional
from pymongo import MongoClient
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
        except Exception as e:
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
    files_collection = db.files
    files_collection.create_index("file_name", unique=True)
    files_collection.create_index("status")
    files_collection.create_index("openai_file_id", sparse=True)
    
    # Batch requests collection
    batch_collection = db.batch_requests
    batch_collection.create_index("batch_id", unique=True)
    batch_collection.create_index("file_id")
    batch_collection.create_index("status")
    
    # Results collection
    results_collection = db.negation_results
    results_collection.create_index("batch_id")
    results_collection.create_index("custom_id")
    
    # Raw results collection
    raw_results_collection = db.batch_results_raw
    raw_results_collection.create_index("batch_id")
    
    # Processing errors collection
    errors_collection = db.processing_errors
    errors_collection.create_index("batch_id")
    
    logger.info("MongoDB collections and indexes set up successfully")
