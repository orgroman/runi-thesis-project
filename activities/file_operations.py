import logging
from bson import ObjectId
from datetime import datetime
from typing import Optional, List

from temporalio import activity

from models import FileMetadata
from mongodb_async import get_async_database

logger = logging.getLogger(__name__)

@activity.defn
async def get_file_metadata_by_id(file_id: str) -> FileMetadata:
    """
    Get file metadata by MongoDB ID.
    
    Args:
        file_id: MongoDB ID of the file
        
    Returns:
        FileMetadata object
    """
    activity.logger.info(f"Getting file metadata for {file_id}")
    
    try:
        # Get MongoDB database
        db = await get_async_database()
        files_collection = db.openai_files
        
        # Convert string ID to ObjectId
        file_obj_id = ObjectId(file_id)
        file = await files_collection.find_one({"_id": file_obj_id})
        
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

@activity.defn
async def check_files_exist_for_dataframe(dataframe_pickle_path: str) -> tuple:
    """
    Pre-check if all files already exist for a dataframe's JSONL batches.
    
    Args:
        dataframe_pickle_path: Path to the dataframe pickle file
        
    Returns:
        Tuple of (all_exist, file_ids)
    """
    activity.logger.info(f"Checking if files exist for dataframe {dataframe_pickle_path}")
    
    try:
        # Get MongoDB client
        db = await get_async_database()
        
        # Find all JSONL batches for this dataframe
        cursor = db.jsonl_batches.find({
            "source_dataframe": dataframe_pickle_path,
            "status": {"$in": ["created", "registered"]}
        })
        
        # Convert cursor to list
        jsonl_batches = await cursor.to_list(length=None)
        
        if not jsonl_batches:
            activity.logger.info(f"No JSONL batches found for {dataframe_pickle_path}")
            return False, []
            
        activity.logger.info(f"Found {len(jsonl_batches)} JSONL batches for {dataframe_pickle_path}")
        
        # Check if all batches have file IDs
        all_have_files = True
        file_ids = []
        
        for batch in jsonl_batches:
            if not batch.get("file_id"):
                all_have_files = False
                break
                
            # Verify the file exists in the files collection
            file = await db.openai_files.find_one({"_id": ObjectId(batch["file_id"])})
            if not file:
                all_have_files = False
                break
                
            file_ids.append(str(batch["file_id"]))
            
        if all_have_files:
            activity.logger.info(f"All {len(file_ids)} files exist for dataframe {dataframe_pickle_path}")
        else:
            activity.logger.info(f"Not all JSONL batches have files for {dataframe_pickle_path}")
            
        return all_have_files, file_ids
        
    except Exception as e:
        activity.logger.error(f"Error checking for existing files: {str(e)}")
        return False, []
