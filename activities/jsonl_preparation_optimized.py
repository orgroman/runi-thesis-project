import logging
from typing import List, Dict, Any, Tuple
from bson import ObjectId

from temporalio import activity

from mongodb import get_mongo_client
from mongodb_utils import check_existing_files

logger = logging.getLogger(__name__)

@activity.defn
async def check_files_exist_for_dataframe(dataframe_pickle_path: str) -> Tuple[bool, List[str]]:
    """
    Pre-check if all files already exist for a dataframe's JSONL batches.
    This activity can be called before registering files to avoid unnecessary processing.
    
    Args:
        dataframe_pickle_path: Path to the dataframe pickle file
        
    Returns:
        Tuple of (all_exist, file_ids)
    """
    activity.logger.info(f"Checking if files exist for dataframe {dataframe_pickle_path}")
    
    try:
        # Get MongoDB client
        mongo_client = get_mongo_client()
        db = mongo_client.patent_negation
        
        # Find all JSONL batches for this dataframe
        jsonl_batches = list(db.jsonl_batches.find({
            "source_dataframe": dataframe_pickle_path,
            "status": {"$in": ["created", "registered"]}
        }))
        
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
            file = db.openai_files.find_one({"_id": ObjectId(batch["file_id"])})
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
