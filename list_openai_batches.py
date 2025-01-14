from openai import AsyncOpenAI
import os
from pathlib import Path
import json
import asyncio

async def main():
    # Get the list of all OpenAI batches
    from openai import OpenAI
    client = AsyncOpenAI(api_key=os.getenv("THESIS_ALON_OPENAI_API_KEY"))
    batch_response_dir = Path(r'C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train\batch_responses')
    batch_jsons = list(batch_response_dir.rglob('*.json'))
    response_data_futures = []
    batch_ids = []
    for batch_json in batch_jsons:
        # load the json
        with open(batch_json, 'r') as f:
            data = json.load(f)
            batch_id = data["batch_response"]["id"]
            batch_ids.append(batch_id)
            response_data_futures.append(client.batches.retrieve(batch_id))
    responses = await asyncio.gather(*response_data_futures)
    #print("All responses gathered!")
    response_data = {}
    for batch_id, response in zip(batch_ids, responses):
        response_data[batch_id] = response
    response_status = {k:v.status for k,v in response_data.items()}
    response_not_failed = {k: v for k,v in response_status.items() if v != 'failed'}
    response_not_failed_not_p = {k: v for k,v in response_status.items() if v != 'failed' and v != 'in_progress'}
    full_complete_responses = [response_data[k] for k in response_not_failed_not_p.keys()]
        
    #print("All responses loaded!")
    print(f"Batch status: {next(iter(response_status.values()))}")
    batch_obj = next(iter(response_data.values()))
    file_content = await client.files.content(batch_obj.output_file_id)
    jsonl_text = file_content.text
    # Load the jsonl text to a list of dictionaries
    jsonl_lines = jsonl_text.split('\n')
    jsonl_lines = [json.loads(line) for line in jsonl_lines if line]
    
    print(f"Batch ID: {batch_obj.id}")
    # file_id = batch_obj.error_file_id
    
    # # get the error file
    # error_file = await client.files.content(file_id)
    
    # print(error_file.text)
            
if __name__ == "__main__":
    asyncio.run(main())
    