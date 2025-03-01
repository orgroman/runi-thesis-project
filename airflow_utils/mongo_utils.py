import logging
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
from bson import ObjectId

from pymongo import MongoClient

logger = logging.getLogger(__name__)

def get_mongo_client(mongo_uri: str = "mongodb://localhost:27017/") -> MongoClient:
    """Get a MongoDB client instance."""
    return MongoClient(mongo_uri)

def check_existing_files(client: MongoClient, dataframe_path: str) -> Tuple[bool, List[str]]:
    """
    Check if files already exist for a dataframe's JSONL batches.
    
    Args:
        client: MongoDB client
        dataframe_path: Path to the dataframe pickle file
        
    Returns:
        Tuple of (all_exist, file_ids)
    """
    db = client.patent_negation
    
    # Find all JSONL batches for this dataframe
    jsonl_batches = list(db.jsonl_batches.find({
        "source_dataframe": dataframe_path,
        "status": {"$in": ["created", "registered"]}
    }))
    
    if not jsonl_batches:
        logger.info(f"No JSONL batches found for {dataframe_path}")
        return False, []
        
    logger.info(f"Found {len(jsonl_batches)} JSONL batches for {dataframe_path}")
    
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
        
    return all_have_files, file_ids

def get_batch_stats(client: MongoClient) -> Dict[str, Any]:
    """Get statistics on batch processing."""
    db = client.patent_negation
    
    stats = {
        "total_files": db.openai_files.count_documents({}),
        "uploaded_files": db.openai_files.count_documents({"status": "uploaded"}),
        "completed_files": db.openai_files.count_documents({"status": "completed"}),
        "failed_files": db.openai_files.count_documents({"status": "failed"}),
        "total_batches": db.batch_requests.count_documents({}),
        "active_batches": db.batch_requests.count_documents({
            "status": {"$in": ["in_progress", "validating", "finalizing"]}
        }),
        "completed_batches": db.batch_requests.count_documents({"status": "completed"}),
        "failed_batches": db.batch_requests.count_documents({
            "status": {"$in": ["failed", "expired", "cancelled"]}
        }),
        "total_results": db.results.count_documents({})
    }
    
    return stats

def reset_hanging_batches(client: MongoClient, hours_threshold: int = 6) -> int:
    """
    Reset batches that have been stuck in processing for too long.
    
    Args:
        client: MongoDB client
        hours_threshold: Consider batches hanging if unchanged for this many hours
        
    Returns:
        Number of batches reset
    """
    db = client.patent_negation
    cutoff_time = datetime.now() - datetime.timedelta(hours=hours_threshold)
    
    # Find batches that haven't been updated recently
    hanging_batches = list(db.batch_requests.find({
        "status": {"$in": ["in_progress", "validating", "finalizing"]},
        "last_checked": {"$lt": cutoff_time}
    }))
    
    count = 0
    for batch in hanging_batches:
        try:
            # Reset file status
            if batch.get("file_id"):
                db.openai_files.update_one(
                    {"_id": ObjectId(batch["file_id"])},
                    {"$set": {"status": "uploaded", "batch_id": None}}
                )
            
            # Mark batch as handled
            db.batch_requests.update_one(
                {"_id": batch["_id"]},
                {"$set": {
                    "status": "cancelled",
                    "processed_at": datetime.now(),
                    "error": f"Reset due to hanging for more than {hours_threshold} hours"
                }}
            )
            count += 1
            
        except Exception as e:
            logger.error(f"Error resetting hanging batch {batch['_id']}: {str(e)}")
    
    return count
