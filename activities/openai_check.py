import logging
from typing import List, Tuple
from bson import ObjectId

from temporalio import activity

from models import FileMetadata
from mongodb_async import get_async_database

logger = logging.getLogger(__name__)

@activity.defn
async def check_existing_openai_files(jsonl_batch_ids: List[str]) -> Tuple[bool, List[FileMetadata]]:
    """
    Check if files are already uploaded to OpenAI by looking at MongoDB records.
    
    Args:
        jsonl_batch_ids: List of JSONL batch IDs to check
        
    Returns:
        Tuple of (all_exist, file_metadatas)
        - all_exist: True if all files already have OpenAI file IDs
        - file_metadatas: List of FileMetadata objects
    """
    activity.logger.info(f"Checking if {len(jsonl_batch_ids)} files are already uploaded to OpenAI")
    
    try:
        db = await get_async_database()
        jsonl_collection = db.jsonl_batches
        files_collection = db.openai_files
        
        file_metadatas = []
        all_exist = True
        
        for batch_id in jsonl_batch_ids:
            # Get the JSONL batch
            jsonl_batch = await jsonl_collection.find_one({"_id": ObjectId(batch_id)})
            
            if not jsonl_batch:
                activity.logger.warning(f"JSONL batch {batch_id} not found")
                all_exist = False
                continue
                
            # Check if batch has a file_id
            if not jsonl_batch.get("file_id"):
                activity.logger.info(f"JSONL batch {batch_id} has no file_id")
                all_exist = False
                continue
                
            # Get the file metadata
            file = await files_collection.find_one({"_id": ObjectId(jsonl_batch["file_id"])})
            
            if not file:
                activity.logger.warning(f"File {jsonl_batch['file_id']} not found for batch {batch_id}")
                all_exist = False
                continue
                
            # Check if file has an OpenAI file ID
            if not file.get("openai_file_id") or file.get("status") != "uploaded":
                activity.logger.info(f"File {jsonl_batch['file_id']} needs to be uploaded to OpenAI")
                all_exist = False
                
            # Create FileMetadata object
            file_metadata = FileMetadata.model_validate(file)
            file_metadata.mongodb_id = str(file["_id"])
            file_metadatas.append(file_metadata)
            
        if all_exist and file_metadatas:
            activity.logger.info(f"All {len(file_metadatas)} files are already uploaded to OpenAI")
        else:
            activity.logger.info(f"Found {len(file_metadatas)} files, but {all_exist=}")
            
        return all_exist, file_metadatas
        
    except Exception as e:
        activity.logger.error(f"Error checking existing OpenAI files: {str(e)}")
        return False, []
