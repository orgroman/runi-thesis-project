import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from temporalio import activity

from models import FileMetadata
from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

@activity.defn
async def register_file_in_mongodb(file_path: str) -> FileMetadata:
    """
    Register a JSONL file in MongoDB for tracking.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        FileMetadata object with MongoDB document ID
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For MongoDB connection or insertion errors
    """
    activity.logger.info(f"Registering file {file_path} in MongoDB")
    
    # Check if file exists
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Get file stats
        file_size = path.stat().st_size
        file_name = path.name
        
        # Create file metadata
        file_metadata = FileMetadata(
            file_path=file_path,
            file_name=file_name,
            file_size=file_size,
            created_at=datetime.now(),
            status="ready",
            attempts=0,
        )
        
        # Get MongoDB client and insert document
        client = get_mongo_client()
        db = client.patent_negation
        files_collection = db.files
        
        result = files_collection.insert_one(file_metadata.model_dump())
        file_metadata.mongodb_id = str(result.inserted_id)
        
        activity.logger.info(f"File registered in MongoDB with ID: {file_metadata.mongodb_id}")
        return file_metadata
        
    except Exception as e:
        activity.logger.error(f"Error registering file in MongoDB: {str(e)}")
        raise
