import logging
from bson import ObjectId
from datetime import datetime
from typing import Optional

from temporalio import activity

from models import FileMetadata
from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

@activity.defn
async def get_file_metadata_by_id(file_id: str) -> FileMetadata:
    """
    Get file metadata by MongoDB ID.
    
    Args:
        file_id: MongoDB ID of the file
        
    Returns:
        FileMetadata object
        
    Raises:
        KeyError: If the file does not exist
        Exception: For MongoDB errors
    """
    activity.logger.info(f"Getting file metadata for {file_id}")
    
    try:
        # Get MongoDB client
        client = get_mongo_client()
        db = client.patent_negation
        files_collection = db.openai_files
        
        # Convert string ID to ObjectId
        file_obj_id = ObjectId(file_id)
        file = files_collection.find_one({"_id": file_obj_id})
        
        if not file:
            raise KeyError(f"File with ID {file_id} not found in MongoDB")
        
        # Create file metadata object
        file_metadata = FileMetadata(
            file_name=file["file_name"],
            file_size=file["file_size"],
            created_at=file.get("created_at", datetime.now()),
            status=file["status"],
            attempts=file.get("attempts", 0),
            jsonl_batch_id=file.get("jsonl_batch_id"),
            mongodb_id=str(file["_id"]),
            openai_file_id=file.get("openai_file_id"),
            uploaded_at=file.get("uploaded_at")
        )
        
        activity.logger.info(f"Successfully retrieved file metadata for {file_id}")
        return file_metadata
        
    except Exception as e:
        activity.logger.error(f"Error getting file metadata: {str(e)}")
        raise
