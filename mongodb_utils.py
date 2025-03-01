import logging
from typing import List, Dict, Optional, Tuple
from bson import ObjectId

from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

async def check_existing_files(dataframe_pickle_path: str) -> Tuple[bool, List[str]]:
    """
    Check if all JSONL batches from a source dataframe already have corresponding files.
    
    Args:
        dataframe_pickle_path: Path to the source dataframe pickle file
        
    Returns:
        Tuple of (all_files_exist, existing_file_ids)
        - all_files_exist: True if all batches have corresponding files
        - existing_file_ids: List of file IDs for batches that have files
    """
    logger.info(f"Checking if all files already exist for {dataframe_pickle_path}")
    
    try:
        # Get MongoDB client
        mongo_client = get_mongo_client()
        db = mongo_client.patent_negation
        
        # Get all JSONL batches for this dataframe
        jsonl_batches = list(db.jsonl_batches.find({
            "source_dataframe": dataframe_pickle_path,
            "status": {"$in": ["created", "registered"]}
        }))
        
        if not jsonl_batches:
            logger.info(f"No JSONL batches found for {dataframe_pickle_path}")
            return False, []
        
        logger.info(f"Found {len(jsonl_batches)} JSONL batches for {dataframe_pickle_path}")
        
        # Check if all batches have corresponding files
        all_file_ids = []
        for batch in jsonl_batches:
            if not batch.get("file_id"):
                logger.info(f"Batch {batch['_id']} has no file_id")
                return False, all_file_ids
            
            # Check if the file exists in the openai_files collection
            file = db.openai_files.find_one({"_id": ObjectId(batch["file_id"])})
            if not file:
                logger.info(f"File {batch['file_id']} not found for batch {batch['_id']}")
                return False, all_file_ids
            
            all_file_ids.append(batch["file_id"])
        
        logger.info(f"All {len(all_file_ids)} files exist for {dataframe_pickle_path}")
        return True, all_file_ids
        
    except Exception as e:
        logger.error(f"Error checking existing files: {str(e)}")
        return False, []
