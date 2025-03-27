import pandas as pd
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def create_batch_input_file(df: pd.DataFrame) -> str:
    """
    Create a JSONL file with batch requests for OpenAI's Batch API.
    Each line contains a request to flip negation in text.
    """
    batch_dir = Path("batch_files")
    batch_dir.mkdir(exist_ok=True)
    
    input_file_path = batch_dir / "negation_flip_input.jsonl"
    
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
    
    return str(input_file_path)

def process_with_batch_api(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe using OpenAI's Batch API and add negation-flipped texts."""
    df_results = df.copy()
    df_results['text_a_flipped'] = ""
    df_results['text_b_flipped'] = ""
    
    print("Preparing batch input file...")
    input_file_path = create_batch_input_file(df_results)
    
    print(f"Uploading batch input file: {input_file_path}")
    with open(input_file_path, "rb") as file:
        batch_file = client.files.create(
            file=file,
            purpose="batch"
        )
    
    print(f"Creating batch with file ID: {batch_file.id}")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    batch_id = batch.id
    print(f"Batch created with ID: {batch_id}")
    
    # Monitor batch status
    status = batch.status
    while status not in ["completed", "failed", "expired", "cancelled"]:
        print(f"Batch status: {status}. Waiting 60 seconds before checking again...")
        time.sleep(60)
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        if status == "in_progress":
            print(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total} requests completed")
    
    if status != "completed":
        print(f"Batch did not complete successfully. Status: {status}")
        return df_results
    
    print("Batch completed! Retrieving results...")
    output_file_id = batch.output_file_id
    
    if output_file_id:
        # Download the results
        output_file_content = client.files.content(output_file_id)
        output_path = Path("batch_files") / "negation_flip_output.jsonl"
        
        # Save the results to a file
        with open(output_path, "wb") as f:
            f.write(output_file_content.read())
        
        # Process the results
        print(f"Processing batch results from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                custom_id = result["custom_id"]
                
                # Check if there was an error
                if result["error"]:
                    print(f"Error for {custom_id}: {result['error']}")
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
                except Exception as e:
                    print(f"Error processing result for {custom_id}: {e}")
    
    # Check for error file
    if batch.error_file_id:
        print(f"Some requests had errors. Error file ID: {batch.error_file_id}")
        # You could download and process the error file here as well
    
    return df_results

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\orgrd\workspace\data\patentmatch_test\test_no_claims_with_neg_final.csv")
    # filter the df where either text_a_response_negation_present or text_b_response_negation_present is True
    df_neg = df[(df['text_a_response_negation_present'] == True) | (df['text_b_response_negation_present'] == True)]
    
    print(f"Number of samples with negation: {len(df_neg)}")
    
    # Process the samples with negation using Batch API
    df_with_flipped = process_with_batch_api(df_neg)
    
    # Save the results to a new CSV file
    output_path = r"C:\Users\orgrd\workspace\data\patentmatch_test\test_no_claims_with_neg_flipped.csv"
    df_with_flipped.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")
    print("Running main function completed")
