import logging
from datetime import datetime
from typing import Optional, List
from bson import ObjectId

from models import FileMetadata, JsonlBatch
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
import os

logger = logging.getLogger(__name__)

# MongoDB connection parameters
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://user:pass@localhost:27017/")
DB_NAME = os.environ.get("MONGO_DB", "patent_negation")

# Global client instance
_async_mongo_client: Optional[AsyncIOMotorClient] = None

async def get_async_mongo_client() -> AsyncIOMotorClient:
    """
    Get an async MongoDB client instance (singleton pattern).
    
    Returns:
        AsyncIOMotorClient instance
    """
    global _async_mongo_client
    
    if _async_mongo_client is None:
        try:
            _async_mongo_client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            await _async_mongo_client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB at {MONGO_URI} (async)")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
            
    return _async_mongo_client

async def get_async_database() -> AsyncIOMotorDatabase:
    """
    Get the async MongoDB database instance.
    
    Returns:
        AsyncIOMotorDatabase instance
    """
    client = await get_async_mongo_client()
    return client[DB_NAME]

async def setup_async_collections():
    """
    Set up MongoDB collections with indexes for better query performance.
    """
    db = await get_async_database()
    
    # Get current collection names
    collection_names = await db.list_collection_names()
    
    # OpenAI Files collection
    if "openai_files" not in collection_names:
        await db.create_collection("openai_files")
        await db.openai_files.create_index("openai_file_id")
        await db.openai_files.create_index("status")
        await db.openai_files.create_index("jsonl_batch_id")
    
    # Other collections setup
    # ... [omitted for brevity]
    
    logger.info("MongoDB collections and indexes set up successfully (async)")

async def register_file_in_mongodb_async(jsonl_batch_id: str) -> FileMetadata:
    """
    Register a JSONL batch from MongoDB as a file for OpenAI processing.
    This is a native Python async function using Motor directly.
    
    Args:
        jsonl_batch_id: MongoDB ID of the JSONL batch
        
    Returns:
        FileMetadata object with MongoDB document ID
        
    Raises:
        KeyError: If the JSONL batch does not exist
        Exception: For MongoDB connection or insertion errors
    """
    logger.info(f"Registering JSONL batch {jsonl_batch_id} in MongoDB")
    
    try:
        # Get MongoDB database and collections
        db = await get_async_database()
        jsonl_collection = db.jsonl_batches
        files_collection = db.openai_files
        
        # Convert string ID to ObjectId
        batch_obj_id = ObjectId(jsonl_batch_id)
        jsonl_batch = await jsonl_collection.find_one({"_id": batch_obj_id})
        
        if not jsonl_batch:
            raise KeyError(f"JSONL batch with ID {jsonl_batch_id} not found in MongoDB")
        
        # Check if batch is already registered with a file
        if jsonl_batch.get("file_id") and jsonl_batch.get("status") == "registered":
            logger.info(f"JSONL batch {jsonl_batch_id} is already registered with file ID {jsonl_batch['file_id']}")
            
            # Retrieve the existing file metadata
            file = await files_collection.find_one({"_id": ObjectId(jsonl_batch["file_id"])})
            
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
        
        # Create base file name
        base_file_name = f"batch_{jsonl_batch['batch_number']}.jsonl"
        
        # Check if a file with this name already exists
        existing_file = await files_collection.find_one({"file_name": base_file_name})
        
        if existing_file:
            # If file exists but is not associated with this batch, create a unique name
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"batch_{jsonl_batch['batch_number']}_{timestamp}.jsonl"
            logger.info(f"File name {base_file_name} already exists, using {file_name} instead")
        else:
            file_name = base_file_name
        
        # Create file metadata
        file_metadata = FileMetadata(
            file_name=file_name,
            file_size=content_size,
            created_at=datetime.now(),
            status="ready",
            attempts=0,
            jsonl_batch_id=str(jsonl_batch["_id"])
        )
        
        # Insert file metadata into MongoDB
        result = await files_collection.insert_one(file_metadata.model_dump(exclude={"mongodb_id"}))
        file_metadata.mongodb_id = str(result.inserted_id)
        
        # Update JSONL batch with file reference
        await jsonl_collection.update_one(
            {"_id": jsonl_batch["_id"]},
            {"$set": {"status": "registered", "file_id": file_metadata.mongodb_id}}
        )
        
        logger.info(f"JSONL batch registered as file with ID: {file_metadata.mongodb_id}")
        return file_metadata
        
    except Exception as e:
        logger.error(f"Error registering JSONL batch in MongoDB: {str(e)}")
        raise

async def register_files_batch_async(jsonl_batch_ids: List[str], concurrency_limit: int = 10) -> List[FileMetadata]:
    """
    Register multiple JSONL batches concurrently using Python's asyncio.
    This performs concurrent operations using Python's native asyncio capabilities.
    
    Args:
        jsonl_batch_ids: List of MongoDB IDs for JSONL batches
        concurrency_limit: Maximum number of concurrent registrations
        
    Returns:
        List of FileMetadata objects that were successfully registered
    """
    import asyncio
    
    logger.info(f"Registering {len(jsonl_batch_ids)} JSONL batches with concurrency {concurrency_limit}")
    file_metadatas = []
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def register_with_semaphore(batch_id):
        async with semaphore:
            try:
                return await register_file_in_mongodb_async(batch_id)
            except Exception as e:
                logger.error(f"Failed to register batch {batch_id}: {str(e)}")
                return None
    
    # Create tasks for all batch IDs
    tasks = [register_with_semaphore(batch_id) for batch_id in jsonl_batch_ids]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Filter out None values (failed registrations)
    file_metadatas = [result for result in results if result is not None]
    
    logger.info(f"Successfully registered {len(file_metadatas)} of {len(jsonl_batch_ids)} JSONL batches")
    return file_metadatas
