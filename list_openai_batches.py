from openai import AsyncOpenAI
import os
from pathlib import Path
import json
import asyncio

async def main():
    # Get the list of all OpenAI batches
    from openai import OpenAI
    client = AsyncOpenAI(api_key=os.getenv("THESIS_ALON_OPENAI_API_KEY"))
    batch_response_dir = Path(r'C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_test\batch_responses')
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
    print("All responses gathered!")
    response_data = {}
    for batch_id, response in zip(batch_ids, responses):
        response_data[batch_id] = response
    response_status = {k:v.status for k,v in response_data.items()}
    response_not_failed = {k: v for k,v in response_status.items() if v != 'failed'}
    response_not_failed_not_p = {k: v for k,v in response_status.items() if v != 'failed' and v != 'in_progress'}
    print("All responses loaded!")
        
if __name__ == "__main__":
    asyncio.run(main())
    