import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field
from temporalio import activity

from models import NegationResponse

logger = logging.getLogger(__name__)

@activity.defn
async def prepare_jsonl_files(dataframe_pickle_path: str, batch_size: int = 1000) -> List[str]:
    """
    Prepare JSONL files for OpenAI batch processing.
    
    Args:
        dataframe_pickle_path: Path to pickled DataFrame
        batch_size: Number of records per batch
        
    Returns:
        List of paths to generated JSONL files
        
    Raises:
        FileNotFoundError: If the pickle file does not exist
        Exception: For JSON serialization errors
    """
    activity.logger.info(f"Preparing JSONL files from {dataframe_pickle_path}")
    
    # Check if pickle file exists
    if not os.path.exists(dataframe_pickle_path):
        raise FileNotFoundError(f"Pickle file not found: {dataframe_pickle_path}")
    
    try:
        # Load DataFrame from pickle
        with open(dataframe_pickle_path, 'rb') as f:
            df = pickle.load(f)
        
        # Create output directory
        output_dir = Path('/c:/Users/orgrd/workspace/repos/runi-thesis-project/output_jsonl')
        output_dir.mkdir(exist_ok=True)
        
        # Process in batches
        num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
        output_files = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            activity.logger.info(f"Processing batch {i + 1}/{num_batches}")
            
            # Create JSONL file path
            output_path = output_dir / f"batch_{i}.jsonl"
            output_files.append(str(output_path))
            
            # Write JSONL file
            with open(output_path, "w", encoding='utf-8') as f:
                for _, row in batch_df.iterrows():
                    jsonl_line = create_jsonl_line(row)
                    f.write(json.dumps(jsonl_line, ensure_ascii=False) + "\n")
        
        activity.logger.info(f"Created {len(output_files)} JSONL files")
        return output_files
        
    except Exception as e:
        activity.logger.error(f"Error preparing JSONL files: {str(e)}")
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
    
    # Define the body with proper schema
    body = {
        "model": "gpt-4-turbo-preview",
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
    
    # Create the full JSONL line object
    return {
        "custom_id": f'request_text_{row["patent_application_id"]}_{row.get("index", "")}',
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }
