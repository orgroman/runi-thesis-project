import logging
from datetime import datetime
from bson import ObjectId

from temporalio import activity

from models import FileMetadata
from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

@activity.defn
async def register_file_in_mongodb(jsonl_batch_id: str) -> FileMetadata:
    """
    Register a JSONL batch from MongoDB as a file for OpenAI processing.
    If already registered, return the existing file metadata.
    
    Args:
        jsonl_batch_id: MongoDB ID of the JSONL batch
        
    Returns:
        FileMetadata object with MongoDB document ID
        
    Raises:
        KeyError: If the JSONL batch does not exist
        Exception: For MongoDB connection or insertion errors
    """
    activity.logger.info(f"Registering JSONL batch {jsonl_batch_id} in MongoDB")
    
    try:
        # Get MongoDB client and retrieve the JSONL batch
        client = get_mongo_client()
        db = client.patent_negation
        jsonl_collection = db.jsonl_batches
        files_collection = db.files
        
        # Convert string ID to ObjectId
        batch_obj_id = ObjectId(jsonl_batch_id)
        jsonl_batch = jsonl_collection.find_one({"_id": batch_obj_id})
        
        if not jsonl_batch:
            raise KeyError(f"JSONL batch with ID {jsonl_batch_id} not found in MongoDB")
        
        # Check if batch is already registered with a file
        if jsonl_batch.get("file_id") and jsonl_batch.get("status") == "registered":
            activity.logger.info(f"JSONL batch {jsonl_batch_id} is already registered with file ID {jsonl_batch['file_id']}")
            
            # Retrieve the existing file metadata
            file = files_collection.find_one({"_id": ObjectId(jsonl_batch["file_id"])})
            
            if file:
                file_metadata = FileMetadata(
                    file_name=file["file_name"],
                    file_size=file["file_size"],
                    created_at=file.get("created_at", datetime.now()),
                    status=file["status"],
                    attempts=file.get("attempts", 0),
                    jsonl_batch_id=jsonl_batch_id,
                    mongodb_id=str(file["_id"]),
                    openai_file_id=file.get("openai_file_id"),
                    uploaded_at=file.get("uploaded_at")
                )
                return file_metadata
        
        # If not registered, get JSONL content size
        content_size = len(jsonl_batch["content"])
        
        # Create file metadata
        file_metadata = FileMetadata(
            file_name=f"batch_{jsonl_batch['batch_number']}.jsonl",
            file_size=content_size,
            created_at=datetime.now(),
            status="ready",
            attempts=0,
            jsonl_batch_id=str(jsonl_batch["_id"])
        )
        
        # Insert file metadata into MongoDB
        result = files_collection.insert_one(file_metadata.model_dump(exclude={"mongodb_id"}))
        file_metadata.mongodb_id = str(result.inserted_id)
        
        # Update JSONL batch with file reference
        jsonl_collection.update_one(
            {"_id": jsonl_batch["_id"]},
            {"$set": {"status": "registered", "file_id": file_metadata.mongodb_id}}
        )
        
        activity.logger.info(f"JSONL batch registered as file with ID: {file_metadata.mongodb_id}")
        return file_metadata
        
    except Exception as e:
        activity.logger.error(f"Error registering JSONL batch in MongoDB: {str(e)}")
        raise
