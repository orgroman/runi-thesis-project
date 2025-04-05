import pandas as pd
import os
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Dict, Any, List, Optional

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Set up MongoDB client
mongodb_client = AsyncIOMotorClient("mongodb://user:pass@localhost:27017")
db = mongodb_client.negation_flip_db

async def save_to_mongodb(collection_name: str, data: Dict[str, Any], identifier: str = None) -> str:
    """
    Save data to MongoDB with timestamp and optional identifier.
    Returns the inserted document's ID.
    """
    collection = db[collection_name]
    
    # Add metadata
    document = {
        "data": data,
        "timestamp": datetime.now(),
        "identifier": identifier
    }
    
    result = await collection.insert_one(document)
    return str(result.inserted_id)

def create_batch_input_file(df: pd.DataFrame) -> str:
    """
    Create a JSONL file with batch requests for OpenAI's Batch API.
    Each line contains a request to flip negation in text.
    """
    batch_dir = Path("batch_files")
    batch_dir.mkdir(exist_ok=True)
    
    input_file_path = batch_dir / f"negation_flip_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    # Store all requests to save to MongoDB later
    batch_requests = []
    
    with open(input_file_path, 'w', encoding='utf-8') as f:
        # Process text_a where negation is present
        mask_a = df['text_a_response_negation_present'] == True
        texts_a = df[mask_a]['text_a']
        
        for i, (idx, text) in enumerate(texts_a.items()):
            request = {
                "custom_id": f"text_a_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that removes negation from text while preserving its meaning as much as possible."},
                        {"role": "user", "content": f"""Remove the negation from the following text while preserving its meaning as much as possible.
                        For example:
                        - "I don't like apples" -> "I like apples"
                        - "She is not happy" -> "She is happy"
                        - "The product does not have these features" -> "The product has these features"
                        
                        Text: {text}
                        
                        Text without negation:"""}
                    ],
                    "max_tokens": 150,
                    "temperature": 0.1
                }
            }
            f.write(json.dumps(request) + '\n')
            batch_requests.append(request)
        
        # Process text_b where negation is present
        mask_b = df['text_b_response_negation_present'] == True
        texts_b = df[mask_b]['text_b']
        
        for i, (idx, text) in enumerate(texts_b.items()):
            request = {
                "custom_id": f"text_b_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that removes negation from text while preserving its meaning as much as possible."},
                        {"role": "user", "content": f"""Remove the negation from the following text while preserving its meaning as much as possible.
                        For example:
                        - "I don't like apples" -> "I like apples"
                        - "She is not happy" -> "She is happy"
                        - "The product does not have these features" -> "The product has these features"
                        
                        Text: {text}
                        
                        Text without negation:"""}
                    ],
                    "max_tokens": 150,
                    "temperature": 0.1
                }
            }
            f.write(json.dumps(request) + '\n')
            batch_requests.append(request)
    
    # Return the file path and batch requests for MongoDB saving
    return str(input_file_path), batch_requests

async def poll_batch_status(batch_id: str, initial_wait: int = 30, max_wait: int = 300, max_attempts: int = 50) -> Dict[str, Any]:
    """
    Poll the batch status with exponential backoff.
    
    Args:
        batch_id: The ID of the batch to poll
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        max_attempts: Maximum number of polling attempts
    
    Returns:
        The completed batch object or raises an exception
    """
    wait_time = initial_wait
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            
            # Save batch status to MongoDB
            status_data = {
                "batch_id": batch_id,
                "status": status,
                "attempt": attempt,
                "request_counts": batch.request_counts,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at
            }
            await save_to_mongodb("batch_status_updates", status_data, batch_id)
            
            print(f"Attempt {attempt}: Batch status: {status}")
            if status == "in_progress":
                print(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total} requests completed")
            
            if status in ["completed", "failed", "expired", "cancelled"]:
                return batch
            
            # Wait with exponential backoff, but cap at max_wait
            await asyncio.sleep(min(wait_time, max_wait))
            wait_time = min(wait_time * 1.5, max_wait)  # Exponential backoff
            
        except Exception as e:
            print(f"Error polling batch status: {e}")
            # Wait a bit before retrying
            await asyncio.sleep(wait_time)
    
    raise Exception(f"Maximum polling attempts ({max_attempts}) reached without completion")

async def process_batch_results(batch: Dict, output_path: str, df_results: pd.DataFrame) -> pd.DataFrame:
    """Process batch results and update the dataframe."""
    print("Processing batch results...")
    
    if not batch.output_file_id:
        print("No output file found in batch results")
        return df_results
    
    # Download the results
    output_file_content = client.files.content(batch.output_file_id)
    
    # Save the results to a file
    with open(output_path, "wb") as f:
        content_bytes = output_file_content.read()
        f.write(content_bytes)
        
    # Save raw results to MongoDB
    try:
        results_data = content_bytes.decode('utf-8')
        results_list = [json.loads(line) for line in results_data.splitlines()]
        await save_to_mongodb("batch_results", {"results": results_list}, f"batch_{batch.id}")
    except Exception as e:
        print(f"Error saving results to MongoDB: {e}")
    
    # Process the results
    print(f"Processing batch results from {output_path}")
    results_processed = 0
    errors = 0
    
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            
            # Check if there was an error
            if result.get("error"):
                print(f"Error for {custom_id}: {result['error']}")
                errors += 1
                continue
            
            # Extract the flipped text
            try:
                flipped_text = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                
                # Parse the custom ID to determine if it's text_a or text_b and get the index
                id_parts = custom_id.split("_")
                field = f"text_{id_parts[1]}_flipped"  # text_a_flipped or text_b_flipped
                idx = int(id_parts[2])
                
                # Update the dataframe
                df_results.at[idx, field] = flipped_text
                results_processed += 1
            except Exception as e:
                print(f"Error processing result for {custom_id}: {e}")
                errors += 1
    
    print(f"Processed {results_processed} results with {errors} errors")
    
    # Check for error file
    if batch.error_file_id:
        print(f"Some requests had errors. Error file ID: {batch.error_file_id}")
        error_content = client.files.content(batch.error_file_id)
        error_path = Path("batch_files") / f"negation_flip_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        # Save the error file
        with open(error_path, "wb") as f:
            error_bytes = error_content.read()
            f.write(error_bytes)
        
        # Save errors to MongoDB
        try:
            error_data = error_bytes.decode('utf-8')
            error_list = [json.loads(line) for line in error_data.splitlines()]
            await save_to_mongodb("batch_errors", {"errors": error_list}, f"batch_{batch.id}")
        except Exception as e:
            print(f"Error saving errors to MongoDB: {e}")
    
    return df_results

async def process_with_batch_api(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe using OpenAI's Batch API and add negation-flipped texts."""
    df_results = df.copy()
    df_results['text_a_flipped'] = ""
    df_results['text_b_flipped'] = ""
    
    print("Preparing batch input file...")
    input_file_path, batch_requests = create_batch_input_file(df_results)
    
    # Save batch requests to MongoDB
    batch_requests_id = await save_to_mongodb(
        "batch_requests", 
        {"file_path": input_file_path, "requests": batch_requests},
        f"batch_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"Batch requests saved to MongoDB with ID: {batch_requests_id}")
    print(f"Uploading batch input file: {input_file_path}")
    
    try:
        with open(input_file_path, "rb") as file:
            batch_file = client.files.create(
                file=file,
                purpose="batch"
            )
        
        # Save file upload result to MongoDB
        file_upload_id = await save_to_mongodb(
            "file_uploads", 
            {"file_id": batch_file.id, "file_path": input_file_path},
            f"file_upload_{batch_file.id}"
        )
        
        print(f"Creating batch with file ID: {batch_file.id}")
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Negation flip batch"}
        )
        
        # Save batch creation result to MongoDB
        batch_creation_id = await save_to_mongodb(
            "batch_creations", 
            dict(batch),
            f"batch_creation_{batch.id}"
        )
        
        batch_id = batch.id
        print(f"Batch created with ID: {batch_id} - MongoDB ID: {batch_creation_id}")
        
        # Poll for batch completion
        completed_batch = await poll_batch_status(batch_id)
        
        if completed_batch.status != "completed":
            print(f"Batch did not complete successfully. Status: {completed_batch.status}")
            return df_results
        
        print("Batch completed! Retrieving results...")
        output_path = Path("batch_files") / f"negation_flip_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        # Process the results and update dataframe
        df_results = await process_batch_results(completed_batch, str(output_path), df_results)
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
    
    return df_results

async def main_async():
    """Async main function to run the batch processing."""
    df = pd.read_csv(r"C:\Users\orgrd\workspace\data\patentmatch_test\test_no_claims_with_neg_final.csv")
    
    # Filter the df where either text_a_response_negation_present or text_b_response_negation_present is True
    df_neg = df[(df['text_a_response_negation_present'] == True) | (df['text_b_response_negation_present'] == True)]
    
    print(f"Number of samples with negation: {len(df_neg)}")
    
    # Process the samples with negation using Batch API
    df_with_flipped = await process_with_batch_api(df_neg)
    
    # Save the results to a new CSV file
    output_path = r"C:\Users\orgrd\workspace\data\patentmatch_test\test_no_claims_with_neg_flipped.csv"
    df_with_flipped.to_csv(output_path, index=False)
    
    # Save final results to MongoDB
    results_summary = {
        "total_samples": len(df_neg),
        "samples_with_text_a_negation": df_neg['text_a_response_negation_present'].sum(),
        "samples_with_text_b_negation": df_neg['text_b_response_negation_present'].sum(),
        "output_path": output_path,
        "timestamp": datetime.now().isoformat()
    }
    
    results_id = await save_to_mongodb(
        "final_results", 
        results_summary,
        f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"Results saved to {output_path} and MongoDB with ID: {results_id}")
    print("Running main function completed")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main_async())
