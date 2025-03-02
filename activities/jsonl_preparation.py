import json
import logging
import io
import pickle
import os
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd
from temporalio import activity

from models import JsonlBatch
from mongodb import get_mongo_client

logger = logging.getLogger(__name__)

@activity.defn
async def prepare_jsonl_files(dataframe_pickle_path: str, batch_size: int = 1000) -> List[str]:
    """
    Prepare JSONL batches and save them to MongoDB.
    If batches from the same source dataframe already exist, reuse them.
    
    Args:
        dataframe_pickle_path: Path to pickled DataFrame
        batch_size: Number of records per batch
        
    Returns:
        List of MongoDB IDs for the JSONL batches
        
    Raises:
        FileNotFoundError: If the pickle file does not exist
        Exception: For MongoDB or JSON serialization errors
    """
    activity.logger.info(f"Preparing JSONL batches from {dataframe_pickle_path}")
    
    # Check if pickle file exists
    if not os.path.exists(dataframe_pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {dataframe_pickle_path}")
    
    try:
        # Get MongoDB client
        mongo_client = get_mongo_client()
        db = mongo_client.patent_negation
        jsonl_collection = db.jsonl_batches
        
        # Check if batches for this dataframe already exist
        existing_batches = list(jsonl_collection.find({
            "source_dataframe": dataframe_pickle_path,
            "status": {"$in": ["created", "registered"]}  # Only consider valid batches
        }).sort("batch_number", 1))
        
        if existing_batches:
            activity.logger.info(f"Found {len(existing_batches)} existing JSONL batches for {dataframe_pickle_path}")
            batch_ids = [str(batch["_id"]) for batch in existing_batches]
            return batch_ids
        
        # Load DataFrame from pickle if we need to create new batches
        activity.logger.info(f"No existing batches found. Creating new JSONL batches for {dataframe_pickle_path}")
        with open(dataframe_pickle_path, 'rb') as f:
            df = pickle.load(f)
        
        # Find highest existing batch number to avoid conflicts
        highest_batch = jsonl_collection.find_one(
            sort=[("batch_number", -1)]
        )
        batch_number_start = (highest_batch["batch_number"] + 1) if highest_batch else 0
        
        # Process in batches
        num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
        batch_ids = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            activity.logger.info(f"Processing batch {i + 1}/{num_batches}")
            
            # Create JSONL content in memory
            jsonl_content = io.StringIO()
            for _, row in batch_df.iterrows():
                jsonl_line = create_jsonl_line(row)
                jsonl_content.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")
            
            # Create batch object with incrementing batch number
            jsonl_batch = JsonlBatch(
                batch_number=batch_number_start + i,
                content=jsonl_content.getvalue(),
                record_count=len(batch_df),
                created_at=datetime.now(),
                status="created",
                source_dataframe=dataframe_pickle_path
            )
            
            # Save to MongoDB
            result = jsonl_collection.insert_one(jsonl_batch.model_dump(exclude={"mongodb_id"}))
            jsonl_batch.mongodb_id = str(result.inserted_id)
            batch_ids.append(jsonl_batch.mongodb_id)
            
            # Clear StringIO buffer
            jsonl_content.close()
        
        activity.logger.info(f"Created {len(batch_ids)} JSONL batches in MongoDB")
        return batch_ids
        
    except Exception as e:
        activity.logger.error(f"Error preparing JSONL batches: {str(e)}")
        raise

def create_jsonl_line(row: pd.Series) -> Dict[str, Any]:
    """Create a single JSONL line for OpenAI batch processing."""
    
    # Extract text to analyze
    text = row['text']
    
    # Construct messages for chat completion
    messages = [
        {"role": "system", "content": "Analyze the text for negations and identify their types."},
        {"role": "user", "content": f"Analyze the following text: {text}"}
    ]
    
    # Define the response schema
    response_schema = {
        "type": "object",
        "properties": {
            "negation_present": {
                "type": "boolean",
                "description": "Whether negation is present in the text"
            },
            "negation_types": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Types of negation found (e.g., 'explicit', 'implicit', 'syntactic', etc.)"
            },
            "short_explanation": {
                "type": "string",
                "description": "Brief explanation of negation findings"
            }
        },
        "required": ["negation_present", "short_explanation"]
    }
    
    # Define the body with proper schema
    body = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "schema": response_schema
        },
        "max_tokens": 500
    }
    
    # Create the full JSONL line object
    return {
        "custom_id": f'request_text_{row["patent_application_id"]}_{row.get("index", "")}',
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }
