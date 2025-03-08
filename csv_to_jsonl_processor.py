"""
Process CSV files into JSONL batches for OpenAI batch processing.
Checks if batches already exist in MongoDB before creating new ones.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import asyncio
from typing import List, Dict, Any, Optional

import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from models import NegationResponse
import config
from mongodb_async import check_collection_exists

# Set up logging
logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

async def check_jsonl_batch_exists(collection, batch_index: int) -> Optional[str]:
    """
    Check if a JSONL batch already exists in MongoDB.
    
    Args:
        collection: MongoDB collection
        batch_index: The batch index to check
        
    Returns:
        The MongoDB document ID if found, None otherwise
    """
    batch_key = f"batch_{batch_index}"
    existing_batch = await collection.find_one({"batch_number": batch_index})
    
    if existing_batch:
        logger.info(f"Batch {batch_index} already exists in MongoDB with ID {existing_batch['_id']}")
        return str(existing_batch["_id"])
    
    return None

def create_jsonl_line(row, column):
    """
    Create a JSONL line for OpenAI batch processing.
    
    Args:
        row: DataFrame row
        column: Column name containing text to analyze
        
    Returns:
        Dictionary representing a line in the JSONL file
    """
    text = row[column]
    messages = [
        {"role": "system", "content": "Analyze the text for negations and identify their types."},
        {"role": "user", "content": f"Analyze the following text: {text}"}
    ]
    
    body = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "negation_response",
                "schema": NegationResponse.model_json_schema()
            }
        },
        "max_tokens": 500
    }
    
    # Create a unique ID for the request using available identifiers
    custom_id = f"request_{column}"
    
    if "patent_application_id" in row:
        custom_id += f"_{row['patent_application_id']}"
    
    if "index" in row:
        custom_id += f"_{row['index']}"
    elif isinstance(row.name, int) or isinstance(row.name, str):
        custom_id += f"_{row.name}"
    
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }

async def process_csv_to_jsonl_batches(
    csv_path: str,
    output_dir: str = None,
    batch_size: int = 1000,
    text_column: str = "text_b",
    mongodb_uri: str = None,
    db_name: str = None,
    collection_name: str = None
) -> List[Dict[str, Any]]:
    """
    Process a CSV file into JSONL batches for OpenAI batch processing.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to store JSONL files (optional)
        batch_size: Number of rows per batch
        text_column: Column containing text to analyze
        mongodb_uri: MongoDB URI (optional)
        db_name: MongoDB database name (optional)
        collection_name: MongoDB collection name (optional)
        
    Returns:
        List of dictionaries with metadata about each batch
    """
    # Use default values from config if not specified
    mongodb_uri = mongodb_uri or config.MONGODB_URI
    db_name = db_name or config.DB_NAME
    collection_name = collection_name or config.JSONL_BATCHES_COLLECTION
    
    # Ensure output directory exists
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_dir = str(Path(csv_path).parent / "jsonl_batches")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read CSV into DataFrame
    logger.info(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"CSV file contains {len(df)} rows")
    
    # Setup MongoDB connection if provided
    mongodb_client = None
    collection = None
    if mongodb_uri:
        try:
            mongodb_client = AsyncIOMotorClient(mongodb_uri)
            
            # Check if the collection exists
            collection_exists = await check_collection_exists(
                db_name=db_name, 
                collection_name=collection_name, 
                client=mongodb_client
            )
            
            if not collection_exists:
                logger.info(f"Collection {collection_name} does not exist in database {db_name}. It will be created.")
            else:
                logger.info(f"Collection {collection_name} exists in database {db_name}.")
            
            collection = mongodb_client[db_name][collection_name]
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            logger.warning("Will proceed without MongoDB storage")
    
    # Calculate number of batches
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    logger.info(f"Processing {len(df)} rows in {num_batches} batches of {batch_size}")
    
    # Process batches
    openai_files_metadata = []
    batch_files_list = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        # Check if batch already exists in MongoDB
        batch_id = None
        if collection_exists:
            continue
            # batch_id = await check_jsonl_batch_exists(collection, i)
            
            # if batch_id:
            #     logger.info(f"Using existing batch {i} from MongoDB (ID: {batch_id})")
            #     # Add metadata without creating new files
            #     jsonl_filepath = str(Path(output_dir) / f"batch_{i}.jsonl")
            #     openai_files_metadata.append({
            #         "jsonl_path": jsonl_filepath,
            #         "batch_number": i,
            #         "record_count": len(batch_df),
            #         "mongodb_id": batch_id,
            #         "created_at": datetime.now().isoformat(),
            #         "source_csv": csv_path
            #     })
            #     continue
        
        logger.info(f"Processing batch {i + 1}/{num_batches}")
        
        # Create JSONL lines
        lines = []
        for _, row in batch_df.iterrows():
            lines.append(create_jsonl_line(row, text_column))
        
        # Write batch to disk as JSONL file
        jsonl_filepath = str(Path(output_dir) / f"batch_{i}.jsonl")
        jsonl_content = ""
        with open(jsonl_filepath, "w", encoding='utf-8') as f:
            for line in lines:
                line_json = json.dumps(line, ensure_ascii=False)
                f.write(line_json + "\n")
                jsonl_content += line_json + "\n"
    
        batch_files_list.append({
            "content": jsonl_content,
            "record_count": len(lines)
        })
        
        # Store in MongoDB if connected
        if collection_exists:
            try:
                batch_document = {
                    "batch_number": i,
                    "content": jsonl_content,
                    "record_count": len(lines),
                    "created_at": datetime.now(),
                    "status": "created",
                    "source_dataframe": csv_path
                }
                
                result = await collection.insert_one(batch_document)
                batch_id = str(result.inserted_id)
                logger.info(f"Stored batch {i} in MongoDB with ID {batch_id}")
            except Exception as e:
                logger.error(f"Error storing batch {i} in MongoDB: {str(e)}")
        
        # Record metadata
        openai_files_metadata.append({
            "jsonl_path": jsonl_filepath,
            "batch_number": i,
            "record_count": len(batch_df),
            "mongodb_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "source_csv": csv_path
        })
    
    # insert many into MongoDB
    collection = mongodb_client[db_name][collection_name]
    try:
        logger.info(f"Storing all batches in MongoDB")
        result = await collection.insert_many(batch_files_list)
        logger.info(f"Stored all batches in MongoDB with IDs {result.inserted_ids}")
    except Exception as e:
        logger.error(f"Error storing all batches in MongoDB: {str(e)}")

    # Write metadata to disk
    metadata_path = str(Path(output_dir) / "openai_files_metadata.json")
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(openai_files_metadata, f, indent=2)
    
    logger.info(f"Created {num_batches} JSONL batches. Metadata stored at {metadata_path}")
    
    # Show sample output from the first batch if available
    first_batch_path = Path(output_dir) / "batch_0.jsonl"
    if first_batch_path.exists():
        logger.info("\nSample output from first batch:")
        with open(first_batch_path, "r", encoding='utf-8') as f:
            logger.info(f.readline())
    
    return openai_files_metadata

async def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process CSV files into JSONL batches for OpenAI")
    #parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("--output-dir", help="Directory to store JSONL files")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of rows per batch")
    parser.add_argument("--text-column", default="text_b", help="Column containing text to analyze")
    parser.add_argument("--mongodb-uri", help="MongoDB URI")
    parser.add_argument("--db-name", help="MongoDB database name")
    parser.add_argument("--collection-name", help="MongoDB collection name")
    
    args = parser.parse_args()
    
    await process_csv_to_jsonl_batches(
        csv_path=r'C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_test_no_claims.csv',
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        text_column=args.text_column,
        mongodb_uri=args.mongodb_uri,
        db_name=args.db_name,
        collection_name=args.collection_name
    )

if __name__ == "__main__":
    asyncio.run(main())
